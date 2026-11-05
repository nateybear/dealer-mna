using DealerMnA
using .Demand
using DataFrames

function (@main)(_)
    snakemake!()
    with_logging() do
        rd = JLD2.load_object(snakemake.input["residual_demand"])
        kept_vks = rd.kept_vks
        K = length(kept_vks)
        ncols = K * (K + 1)

        price_table = read_arrow(snakemake.input["price_table"])
        vk_to_group = Dict{VehicleKey, Tuple{String, Int64, String}}()
        for row in eachrow(price_table)
            vk = (row.make, row.model, row.model_year, Float32(row.trim_rank))
            vk_to_group[vk] = (row.model, row.model_year, row.extra)
        end

        AccKey = Tuple{String, Int64, String, String}
        sum_eta = Dict{AccKey, Float64}()
        sum_sq = Dict{AccKey, Float64}()
        n_by_key = Dict{AccKey, Int}()

        sim_paths = snakemake.input["simulations"]
        all_ts = Set{QuarterKey}()
        for path in sim_paths
            sim = JLD2.load_object(path)
            for ((t, _f), _) in sim
                push!(all_ts, t)
            end
            break
        end
        t0 = minimum(all_ts)

        n_rows = 0
        for path in sim_paths
            sim = JLD2.load_object(path)
            for ((t, _f), nt) in sim
                t == t0 && continue
                isempty(nt.state_expected) && continue
                length(nt.state_actual) == ncols || continue
                length(nt.state_expected) == ncols || continue
                n_rows += 1
                for c in 1:ncols
                    a = nt.state_actual[c]
                    e = nt.state_expected[c]
                    (isnan(a) || isnan(e)) && continue
                    j_focal = ((c - 1) % K) + 1
                    k_col = ((c - 1) ÷ K) + 1
                    coef_type = if k_col == 1
                        "intercept"
                    elseif k_col - 1 == j_focal
                        "own"
                    else
                        "cross"
                    end
                    vk_focal = kept_vks[j_focal]
                    haskey(vk_to_group, vk_focal) || continue
                    grp = vk_to_group[vk_focal]
                    key = (grp[1], grp[2], grp[3], coef_type)
                    eta = Float64(a - e)
                    sum_eta[key] = get(sum_eta, key, 0.0) + eta
                    sum_sq[key] = get(sum_sq, key, 0.0) + eta * eta
                    n_by_key[key] = get(n_by_key, key, 0) + 1
                end
            end
        end
        @info "Processed sim (t, f) rows" n_rows

        out = DataFrame(
            focal_model = String[], focal_my = Int64[], focal_extra = String[],
            coef_type = String[],
            mean_eta = Float64[], std_eta = Float64[], n = Int[],
        )
        for (key, ne) in n_by_key
            fm, my, extra, ct = key
            m = sum_eta[key] / ne
            v = (sum_sq[key] / ne) - m^2
            push!(out, (fm, my, extra, ct, m, sqrt(max(v, 0.0)), ne))
        end
        sort!(out, [:coef_type, :focal_my, :focal_model, :focal_extra])
        Arrow.write(snakemake.output["summary"], out)
        @info "By-VK anticipation summary" n_rows = nrow(out)
    end
    return nothing
end
