using DealerMnA
using .Demand
using DataFrames
using Statistics: mean, std

function (@main)(_)
    snakemake!()
    with_logging() do
        rd = JLD2.load_object(snakemake.input["residual_demand"])
        obs_matrix = rd.matrix
        kept_vks = rd.kept_vks
        tf_keys = rd.tf_keys
        tf_to_row = Dict(k => i for (i, k) in enumerate(tf_keys))
        K = length(kept_vks)
        ncols = K * (K + 1)
        @info "Loaded obs state matrix" K M = length(tf_keys) ncols

        price_table = read_arrow(snakemake.input["price_table"])
        vk_to_group = Dict{VehicleKey, Tuple{String, Int64, String}}()
        for row in eachrow(price_table)
            vk = (row.make, row.model, row.model_year, Float32(row.trim_rank))
            vk_to_group[vk] = (row.model, row.model_year, row.extra)
        end

        sum_diff = zeros(Float64, ncols)
        sum_abs_diff = zeros(Float64, ncols)
        sum_sim = zeros(Float64, ncols)
        sum_obs = zeros(Float64, ncols)
        n_finite = zeros(Int, ncols)
        n_nan_sim_only = zeros(Int, ncols)
        n_nan_obs_only = zeros(Int, ncols)
        n_both_nan = zeros(Int, ncols)

        all_quarters = Set{QuarterKey}(t for (t, _) in tf_keys)
        q_sum_abs_diff = Dict(q => zeros(Float64, ncols) for q in all_quarters)
        q_n_finite = Dict(q => zeros(Int, ncols) for q in all_quarters)

        sim_paths = snakemake.input["simulations"]
        n_seeds_processed = 0
        n_rows = 0
        n_skipped_empty = 0
        n_skipped_seed = 0
        all_ts = Set{QuarterKey}()
        for path in sim_paths
            sim = JLD2.load_object(path)
            for ((t, f), nt) in sim
                push!(all_ts, t)
            end
            break
        end
        t0 = minimum(all_ts)

        for path in sim_paths
            sim = JLD2.load_object(path)
            for ((t, f), nt) in sim
                if t == t0
                    n_skipped_seed += 1
                    continue
                end
                if isempty(nt.assortment)
                    n_skipped_empty += 1
                    continue
                end
                length(nt.state_actual) == ncols || continue
                haskey(tf_to_row, (t, f)) || continue
                obs_row = view(obs_matrix, tf_to_row[(t, f)], :)
                sim_state = nt.state_actual
                for c in 1:ncols
                    s = sim_state[c]
                    o = obs_row[c]
                    if isnan(s) && isnan(o)
                        n_both_nan[c] += 1
                    elseif isnan(s)
                        n_nan_sim_only[c] += 1
                    elseif isnan(o)
                        n_nan_obs_only[c] += 1
                    else
                        d = Float64(s - o)
                        sum_diff[c] += d
                        sum_abs_diff[c] += abs(d)
                        sum_sim[c] += Float64(s)
                        sum_obs[c] += Float64(o)
                        n_finite[c] += 1
                        q_sum_abs_diff[t][c] += abs(d)
                        q_n_finite[t][c] += 1
                    end
                end
                n_rows += 1
            end
            n_seeds_processed += 1
        end
        @info "Processed sim data" n_seeds_processed n_rows n_skipped_seed n_skipped_empty

        cols = DataFrame(
            col_idx = Int[],
            focal_model = String[], focal_model_year = Int[], focal_extra = String[],
            coef_type = String[],
            partner_model = String[], partner_model_year = Int[], partner_extra = String[],
            mean_sim = Float64[], mean_obs = Float64[], mean_diff = Float64[],
            mean_abs_diff = Float64[],
            n_finite = Int[], n_nan_sim_only = Int[], n_nan_obs_only = Int[],
            n_both_nan = Int[],
        )
        for c in 1:ncols
            j_focal = ((c - 1) % K) + 1
            k_col = ((c - 1) ÷ K) + 1
            focal_vk = kept_vks[j_focal]
            focal_grp = get(vk_to_group, focal_vk, ("unknown", 0, "unknown"))
            coef_type, partner_vk = if k_col == 1
                ("intercept", nothing)
            elseif k_col - 1 == j_focal
                ("own", nothing)
            else
                ("cross", kept_vks[k_col - 1])
            end
            partner_grp =
                partner_vk === nothing ? ("", 0, "") :
                get(vk_to_group, partner_vk, ("unknown", 0, "unknown"))
            n = n_finite[c]
            push!(
                cols,
                (
                    c, focal_grp[1], focal_grp[2], focal_grp[3],
                    coef_type, partner_grp[1], partner_grp[2], partner_grp[3],
                    n > 0 ? sum_sim[c] / n : NaN,
                    n > 0 ? sum_obs[c] / n : NaN,
                    n > 0 ? sum_diff[c] / n : NaN,
                    n > 0 ? sum_abs_diff[c] / n : NaN,
                    n, n_nan_sim_only[c], n_nan_obs_only[c], n_both_nan[c],
                ),
            )
        end
        Arrow.write(snakemake.output["columns"], cols)
        @info "Per-column drift table" n_cols = nrow(cols) n_cols_any_finite = count(
            >(0), cols.n_finite
        )

        summary = DataFrame(
            year = Int[], quarter = Int[], coef_type = String[], focal_model_year = Int[],
            n_cols = Int[], n_obs_total = Int[],
            mean_abs_diff = Float64[], max_abs_diff = Float64[],
        )
        all_mys = unique(cols.focal_model_year)
        for q in sort(collect(all_quarters))
            for ct in ("intercept", "own", "cross")
                for my in all_mys
                    mask =
                        (cols.coef_type .== ct) .& (cols.focal_model_year .== my)
                    col_idxs = cols.col_idx[mask]
                    isempty(col_idxs) && continue
                    total_abs = 0.0
                    total_n = 0
                    max_mean = 0.0
                    n_cols_q = 0
                    for ci in col_idxs
                        n_q = q_n_finite[q][ci]
                        n_q > 0 || continue
                        n_cols_q += 1
                        m = q_sum_abs_diff[q][ci] / n_q
                        total_abs += m * n_q
                        total_n += n_q
                        max_mean = max(max_mean, m)
                    end
                    total_n > 0 || continue
                    push!(
                        summary,
                        (
                            q[1], q[2], ct, my, n_cols_q, total_n,
                            total_abs / total_n, max_mean,
                        ),
                    )
                end
            end
        end
        sort!(summary, [:coef_type, :focal_model_year, :year, :quarter])
        Arrow.write(snakemake.output["summary"], summary)
        @info "Per-(quarter × coef × focal_my) summary" n_rows = nrow(summary)
    end
    return nothing
end
