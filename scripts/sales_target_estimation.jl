using DealerMnA
using XGBoost
using LoggingExtras
using LinearAlgebra
using LogExpFunctions
using StatsBase: mean, std

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

        # Start the raw-output prediction near mean(q) so (q − f)/τ centers
        # around 0 and the Hessian is healthy. The default base_score = −2.0
        # (appropriate for p_base/ρ̃) leaves π ≈ 1 for every row and the
        # Newton step degenerates — no tree ever splits.
        params = merge(Supply.XGB_DEFAULT_PARAMS, (base_score = Float64(mean(q)),))

        # Link temperature τ = std(q). Without it σ((q − f)/1) saturates on
        # the q scale (range 1 → ~1000), Hessian π(1 − π) ≈ 0 everywhere
        # except a ±5-car window, and split-gain ≈ Σg² / (Σh + λ) collapses
        # to λ-dominated noise — best_round stays at 0.
        τ = Float32(std(q))
        @info "Sales-target link" mean_q = mean(q) τ

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
            max_round = 200,
            patience = 5,
            params = params,
        )
        @info "Sales-target booster" best_round nrows = n mean_p_compliant = mean(ι)

        # Save (booster_bytes, τ): consumers need τ to recover
        # P(ι=1) = σ((q − φ̂)/τ) from the booster's raw φ predictions.
        JLD2.save_object(
            snakemake.output["sales_targets"], (XGBoost.serialize(booster), τ)
        )
    end
    return nothing
end
