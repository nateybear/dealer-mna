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
        K = length(kept_vks)
        @info "Loaded obs state matrix" K M = length(tf_keys)

        # J_obs per quarter: for each (t, f) row, count how many of the K
        # intercepts (flat cols 1..K) are non-NaN — residual_demand only fills
        # rows for products carried at that (t, f), so finite intercept ≡ live
        # product.
        j_obs_by_t = Dict{QuarterKey, Int}()
        n_dq_obs_by_t = Dict{QuarterKey, Int}()
        for (i, (t, _f)) in enumerate(tf_keys)
            n_live = 0
            @inbounds for c in 1:K
                isnan(obs_matrix[i, c]) || (n_live += 1)
            end
            j_obs_by_t[t] = get(j_obs_by_t, t, 0) + n_live
            n_dq_obs_by_t[t] = get(n_dq_obs_by_t, t, 0) + 1
        end

        # J_sim per (quarter, seed): sum of assortment sizes over all (t, f)
        # in that seed at that quarter. Also track dealer-quarter count.
        j_sim_by_t_seed = Dict{Tuple{QuarterKey, Int}, Int}()
        n_dq_sim_by_t_seed = Dict{Tuple{QuarterKey, Int}, Int}()
        sim_paths = snakemake.input["simulations"]
        for (seed_idx, path) in enumerate(sim_paths)
            sim = JLD2.load_object(path)
            for ((t, _f), nt) in sim
                k = (t, seed_idx)
                j_sim_by_t_seed[k] = get(j_sim_by_t_seed, k, 0) + length(nt.assortment)
                n_dq_sim_by_t_seed[k] = get(n_dq_sim_by_t_seed, k, 0) + 1
            end
        end
        @info "Loaded sim data" n_seeds = length(sim_paths)

        all_t = union(Set(keys(j_obs_by_t)), Set(k[1] for k in keys(j_sim_by_t_seed)))

        out = DataFrame(
            year = Int[], quarter = Int[],
            j_obs = Int[], n_dq_obs = Int[], mean_n_obs = Float64[],
            j_sim_mean = Float64[], j_sim_std = Float64[],
            n_dq_sim = Float64[], mean_n_sim = Float64[],
            j_ratio = Float64[], log_j_shift = Float64[],
        )
        for t in sort(collect(all_t))
            j_obs = get(j_obs_by_t, t, 0)
            n_dq_obs = get(n_dq_obs_by_t, t, 0)
            sim_vals = [v for ((tt, _), v) in j_sim_by_t_seed if tt == t]
            dq_sim_vals =
                [Float64(v) for ((tt, _), v) in n_dq_sim_by_t_seed if tt == t]
            if isempty(sim_vals)
                push!(
                    out,
                    (
                        t[1], t[2], j_obs, n_dq_obs,
                        n_dq_obs > 0 ? j_obs / n_dq_obs : NaN,
                        NaN, NaN, NaN, NaN, NaN, NaN,
                    ),
                )
                continue
            end
            j_sim_mean = mean(sim_vals)
            j_sim_sd = length(sim_vals) > 1 ? std(sim_vals) : 0.0
            n_dq_sim = mean(dq_sim_vals)
            push!(
                out,
                (
                    t[1], t[2], j_obs, n_dq_obs,
                    n_dq_obs > 0 ? j_obs / n_dq_obs : NaN,
                    j_sim_mean, j_sim_sd, n_dq_sim,
                    n_dq_sim > 0 ? j_sim_mean / n_dq_sim : NaN,
                    j_obs > 0 ? j_sim_mean / j_obs : NaN,
                    (j_obs > 0 && j_sim_mean > 0) ? log(j_sim_mean) - log(j_obs) : NaN,
                ),
            )
        end
        Arrow.write(snakemake.output["summary"], out)
        @info "Per-quarter J_t comparison" n_rows = nrow(out)
    end
    return nothing
end
