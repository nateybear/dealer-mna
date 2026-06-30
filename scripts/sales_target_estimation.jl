using DealerMnA
using XGBoost
using LoggingExtras
using LinearAlgebra
using LogExpFunctions
using StatsBase: mean, std, quantile

function (@main)(_)
    snakemake!()
    with_logging() do
        demand_model = Demand.DemandModel(;
            JLD2.load_object(snakemake.input["model_frames"])...,
        )
        state_matrix = JLD2.load_object(snakemake.input["state_matrix"])
        intermed = Supply.tf_to_j_lookup(demand_model)
        state_lookup = Dict(k => i for (i, k) in enumerate(keys(intermed)))

        compliance = read_arrow(snakemake.input["compliance"])
        fords = read_arrow(snakemake.input["fords"])

        # q_{d,t} = total units sold per dealer-quarter.
        dq = Supply.make_dealer_quarter(fords)
        select!(dq, :dealer_uid, :t, :total_units => :q)

        panel = innerjoin(compliance, dq, on = [:dealer_uid, :t])
        panel.ι = Float32.(panel.p_compliant)

        # Every (dealer, t) in the panel must have a state-matrix entry; error if not.
        unmapped = [
            (r.dealer_uid, r.t) for
            r in eachrow(panel) if !haskey(
                state_lookup, ((Dates.year(r.t), Dates.quarterofyear(r.t)), r.dealer_uid)
            )
        ]
        if !isempty(unmapped)
            error(
                "$(length(unmapped)) (dealer_uid, t) rows in panel are missing from " *
                "the state-matrix lookup. Sample: $(first(unmapped, 5))",
            )
        end

        n = nrow(panel)
        ncols = size(state_matrix, 2)
        X = Matrix{Float32}(undef, n, ncols)
        for (i, r) in enumerate(eachrow(panel))
            quarter = (Dates.year(r.t), Dates.quarterofyear(r.t))
            X[i, :] .= @view state_matrix[state_lookup[(quarter, r.dealer_uid)], :]
        end

        q = Float32.(panel.q)
        ι = panel.ι
        # DMatrix label is decorative — closures recover q and ι. Use the
        # soft-signed quantity (2ι − 1)·q for inspectability.
        y = @. (2 * ι - 1) * q

        # Link temperature τ = std(q) — σ((q − f)/τ) transitions over a ±τ-car
        # window, matching the realistic fuzziness of "made target."
        τ = Float32(std(q))

        # Per-row Hessian is π(1 − π)/τ² ≈ 0.25/τ², so per-leaf ΣH for ~50
        # rows is O(50/τ²) ≈ 1e-3 here. Defaults (λ=1, min_child_weight=5)
        # were tuned for the binary p_base/ρ̃ boosters where per-row H ≈ 0.1;
        # carrying them over makes λ and min_child_weight both swamp ΣH, so
        # nothing splits and the booster collapses to a constant. Scale them
        # down by ~τ², deepen the trees, and ease the learning rate to let
        # the booster move on the q-scale within max_round.
        params = merge(
            Supply.XGB_DEFAULT_PARAMS,
            (
                base_score = Float64(mean(q)),
                max_depth = 6,
                min_child_weight = 1.0f-4,
                lambda = 1.0f-5,
                eta = 0.05,
            ),
        )

        # Stratify the 80/20 OOS split on ι > 0.5 vs ι ≤ 0.5 so val NLL is
        # always computed on a mix of compliers and non-compliers — without
        # this, an unlucky draw of the ~13% compliers can throw the
        # stopping criterion.
        ι_strata = [x > 0.5 ? 1 : 0 for x in ι]

        @info "Sales-target setup" mean_q = mean(q) τ mean_p_compliant = mean(ι) frac_strata_1 =
            mean(ι_strata)

        booster, best_round = Supply.fit_xgb_custom_oos(
            idxs ->
                let m = length(idxs), q_s = q[idxs], ι_s = ι[idxs]
                    ŷ -> begin
                        grad = Vector{Float32}(undef, m)
                        hess = Vector{Float32}(undef, m)
                        @inbounds for j in 1:m
                            grad[j], hess[j] =
                                Supply.phi_grad_hess(ŷ[j], q_s[j], ι_s[j], τ)
                        end
                        return grad, hess
                    end
                end,
            idxs ->
                let m = length(idxs), q_s = q[idxs], ι_s = ι[idxs]
                    ŷ -> begin
                        nll = 0.0
                        @inbounds for j in 1:m
                            nll += Supply.phi_nll(ŷ[j], q_s[j], ι_s[j], τ)
                        end
                        return nll / m
                    end
                end,
            X,
            y;
            max_round = 1000,
            patience = 20,
            params = params,
            strata = ι_strata,
        )
        @info "Sales-target booster" best_round nrows = n

        # Save (booster_bytes, τ): consumers need τ to recover
        # P(ι=1) = σ((q − φ̂)/τ) from the booster's raw φ predictions.
        JLD2.save_object(
            snakemake.output["sales_targets"], (XGBoost.serialize(booster), τ)
        )
    end
    return nothing
end
