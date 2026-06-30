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
            JLD2.load_object(snakemake.input["model_frames"])...
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

        # Log link on φ: booster fits f = log φ, so φ = exp(f) > 0. The
        # chain-rule exp(f) factor in the gradient/Hessian lifts H back to
        # O(1) magnitude, so default-ish λ / min_child_weight work.
        # base_score = log(mean(q)) so initial exp(f) ≈ mean(q) and π
        # centers near 0.5.
        params = merge(
            Supply.XGB_DEFAULT_PARAMS,
            (
                base_score = Float64(log(mean(q))),
                max_depth = 4,
                min_child_weight = 1,
                lambda = 1.0,
                eta = 0.1,
            ),
        )

        # Stratify the 80/20 OOS split on ι > 0.5 vs ι ≤ 0.5 so val NLL is
        # always computed on a mix of compliers and non-compliers — without
        # this, an unlucky draw of the ~13% high-ι rows throws stopping.
        ι_strata = [x > 0.5 ? 1 : 0 for x in ι]

        # τ-dependent gradient/NLL factories. Closures capture τ; we build
        # fresh ones per grid point.
        function make_factories(τ)
            (
                idxs ->
                    let m = length(idxs), q_s = q[idxs], ι_s = ι[idxs]
                        ŷ -> begin
                            grad = Vector{Float32}(undef, m)
                            hess = Vector{Float32}(undef, m)
                            @inbounds for j in 1:m
                                grad[j], hess[j] = Supply.phi_grad_hess(ŷ[j], q_s[j], ι_s[j], τ)
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
            )
        end

        # τ is formally underidentified in the population (can be absorbed
        # into a rescaling of φ), but with a capacity-bounded booster it
        # trades off bias vs variance. Pick τ by OOS val NLL on a geometric
        # grid bracketing std(q) ≈ 157. Failures (best_round == 0, etc.)
        # are caught so a single τ blowing up doesn't kill the search.
        τ_grid = Float32.([160])
        @info "Sales-target setup" mean_q = mean(q) std_q = std(q) mean_p_compliant = mean(
            ι
        ) τ_grid
        grid_results = map(τ_grid) do τ_cand
            try
                gh, nll = make_factories(τ_cand)
                b, br, bv = Supply.fit_xgb_custom_oos(
                    gh,
                    nll,
                    X,
                    y;
                    max_round = 200,
                    patience = 5,
                    params = params,
                    strata = ι_strata,
                )
                br > 0 || @warn "τ produced best_round = 0; skipping" τ_cand
                return (τ = τ_cand, booster = b, best_round = br, best_val = bv)
            catch e
                @warn "τ fit failed; skipping" τ_cand exception = (e, catch_backtrace())
                return (τ = τ_cand, booster = nothing, best_round = 0, best_val = Inf)
            end
        end
        for r in grid_results
            @info "τ grid" τ = r.τ best_round = r.best_round best_val = r.best_val
        end

        valid = filter(r -> r.best_round > 0 && isfinite(r.best_val), grid_results)
        isempty(valid) &&
            error("No τ in $(τ_grid) produced a valid fit. Widen the grid.")
        i = argmin([r.best_val for r in valid])
        best = valid[i]
        τ = best.τ
        booster = best.booster
        best_round = best.best_round
        @info "Picked τ" τ best_round best_val = best.best_val

        # Sanity-check the fitted φ̂ distribution.
        φ̂ = exp.(XGBoost.predict(booster, X))
        φ̂_qs = quantile(φ̂, [0.05, 0.25, 0.5, 0.75, 0.95])
        q_qs = quantile(q, [0.05, 0.25, 0.5, 0.75, 0.95])
        @info "Sales-target φ̂ quantiles (5/25/50/75/95)" φ̂_qs q_qs

        # Save (booster_bytes, τ): the booster's raw output is f = log φ, so
        # consumers recover φ̂ = exp(f̂) and then P(ι=1) = σ((q − φ̂)/τ).
        JLD2.save_object(
            snakemake.output["sales_targets"], (XGBoost.serialize(booster), τ)
        )
    end
    return nothing
end
