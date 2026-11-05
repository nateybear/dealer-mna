using DealerMnA
using .Demand
using XGBoost
using DataFrames
using Statistics: mean
using Random
import Base: size

function (@main)(_)
    snakemake!()
    with_logging() do
        model = DemandModel(; JLD2.load_object(snakemake.input["model_frames"])...)
        rd = JLD2.load_object(snakemake.input["residual_demand"])
        state_matrix = rd.matrix
        kept_vks = rd.kept_vks
        tf_keys = rd.tf_keys
        state_lookup = Dict(k => i for (i, k) in enumerate(tf_keys))
        K = length(kept_vks)
        ncols_state = size(state_matrix, 2)

        kept_set = Set(kept_vks)
        intermed_raw = Supply.tf_to_j_lookup(model)
        intermed = Dict(
            k => VehicleKey[j for j in get(intermed_raw, k, VehicleKey[]) if j ∈ kept_set]
            for k in tf_keys
        )

        price_table = read_arrow(snakemake.input["price_table"])
        vk_to_group = Dict{VehicleKey, Tuple{String, Int64, String}}()
        for row in eachrow(price_table)
            vk = (row.make, row.model, row.model_year, Float32(row.trim_rank))
            vk_to_group[vk] = (row.model, row.model_year, row.extra)
        end

        tlag_of(t) = t[2] == 1 ? (t[1] - 1, 4) : (t[1], t[2] - 1)

        function panel_for_product(j_idx)
            vk = kept_vks[j_idx]
            rows_tlag = Int[]
            for (tf, _) in state_lookup
                vk in intermed[tf] || continue
                tlag = tlag_of(tf[1])
                haskey(state_lookup, (tlag, tf[2])) || continue
                push!(rows_tlag, state_lookup[(tlag, tf[2])])
            end
            return rows_tlag
        end

        function bootup(bytes, ncols)
            isempty(bytes) && return nothing
            dummy = XGBoost.DMatrix(zeros(Float32, 1, ncols))
            b = XGBoost.Booster(dummy)
            XGBoost.load!(b, bytes)
            finalize(dummy)
            return b
        end

        state_transition = JLD2.load_object(snakemake.input["state_transition"])
        median_idx = findfirst(≈(0.5f0), Supply.QUANTILE_LEVELS)

        # Covers training (~7.9–8.4) and sim (~7.6–8.4) log(J_t) ranges.
        log_j_grid = Float32[7.3, 7.5, 7.7, 7.9, 8.1, 8.3, 8.5]
        quarter_grid = Int[1, 2, 3, 4]
        rng = MersenneTwister(20261108)

        out = DataFrame(
            focal_model = String[], focal_my = Int64[], focal_extra = String[],
            quarter = Int[], log_j = Float32[],
            mean_pred_intercept = Float64[], n_samples = Int[],
        )
        for (j_idx, vk) in enumerate(kept_vks)
            haskey(state_transition, vk) || continue
            haskey(vk_to_group, vk) || continue
            grp = vk_to_group[vk]
            int_bytes, _own_bytes, _cross_bytes = state_transition[vk]
            isempty(int_bytes[median_idx]) && continue

            rows_tlag = panel_for_product(j_idx)
            n = length(rows_tlag)
            n == 0 && continue

            n_samples = min(n, 50)
            idxs = n == n_samples ? collect(1:n) : rand(rng, 1:n, n_samples)

            b_median = bootup(int_bytes[median_idx], ncols_state + 2)
            b_median === nothing && continue

            for q in quarter_grid
                for log_j in log_j_grid
                    X = Matrix{Float32}(undef, n_samples, ncols_state + 2)
                    for (r, i) in enumerate(idxs)
                        X[r, 1:ncols_state] .= @view state_matrix[rows_tlag[i], :]
                        X[r, ncols_state + 1] = log_j
                        X[r, ncols_state + 2] = Float32(q)
                    end
                    dm = XGBoost.DMatrix(X)
                    ŷ = XGBoost.predict(b_median, dm; margin = true)
                    finalize(dm)
                    push!(
                        out,
                        (grp[1], grp[2], grp[3], q, log_j, Float64(mean(ŷ)), n_samples),
                    )
                end
            end
            finalize(b_median)
        end
        sort!(out, [:focal_my, :focal_model, :focal_extra, :quarter, :log_j])
        Arrow.write(snakemake.output["response"], out)
        @info "log(J) response curve" n_rows = nrow(out)
    end
    return nothing
end
