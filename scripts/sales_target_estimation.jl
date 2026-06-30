using DealerMnA
using XGBoost
using LoggingExtras
using LinearAlgebra
using LogExpFunctions
using StatsBase: mean

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
        panel.ι = Float32.(panel.p_compliant .> 0.5)

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
        # ±q encoding for the DMatrix label is decorative — closures recover q and ι.
        y = @. sign(2 * ι - 1) * q

        booster, best_round = Supply.fit_xgb_custom_oos(
            idxs ->
                let m = length(idxs), q_s = q[idxs], ι_s = ι[idxs]
                    ŷ -> begin
                        grad = Vector{Float32}(undef, m)
                        hess = Vector{Float32}(undef, m)
                        @inbounds for j in 1:m
                            grad[j], hess[j] = Supply.phi_grad_hess(ŷ[j], q_s[j], ι_s[j])
                        end
                        return grad, hess
                    end
                end,
            idxs ->
                let m = length(idxs), q_s = q[idxs], ι_s = ι[idxs]
                    ŷ -> begin
                        nll = 0.0
                        @inbounds for j in 1:m
                            nll += Supply.phi_nll(ŷ[j], q_s[j], ι_s[j])
                        end
                        return nll / m
                    end
                end,
            X,
            y;
            max_round = 200,
            patience = 5,
        )
        @info "Sales-target booster" best_round nrows = n mean_q = mean(q) frac_compliant = mean(
            ι
        )

        JLD2.save_object(snakemake.output["sales_targets"], XGBoost.serialize(booster))
    end
    return nothing
end
