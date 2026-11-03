using DealerMnA
using .Demand
using XGBoost
using DataFrames
using Statistics: mean
using Dates
import Base: size

function (@main)(_)
    snakemake!()
    with_logging() do
        model = DemandModel(; JLD2.load_object(snakemake.input["model_frames"])...)
        _rd = JLD2.load_object(snakemake.input["residual_demand"])
        state_matrix = _rd.matrix
        kept_vks = _rd.kept_vks
        tf_keys = _rd.tf_keys
        state_lookup = Dict(k => i for (i, k) in enumerate(tf_keys))
        J = length(kept_vks)
        prod_index = Dict(p => i for (i, p) in enumerate(kept_vks))
        kept_set = Set(kept_vks)
        ncols_state = size(state_matrix, 2)
        τs = Supply.QUANTILE_LEVELS

        intermed_raw = Supply.tf_to_j_lookup(model)
        intermed = Dict(
            k => VehicleKey[j for j in get(intermed_raw, k, VehicleKey[]) if j ∈ kept_set]
            for k in tf_keys
        )

        # J_by_t = total kept-VK (t, j, f) triples per quarter, summed across
        # dealers. Matches the log(J_t) feature in state_transition training.
        J_by_t = Dict{QuarterKey, Int}()
        for tf in tf_keys
            J_by_t[tf[1]] = get(J_by_t, tf[1], 0) + length(intermed[tf])
        end

        price_table = read_arrow(snakemake.input["price_table"])
        vk_to_group = Dict{VehicleKey, Tuple{String, Int64, String}}()
        for row in eachrow(price_table)
            vk = (row.make, row.model, row.model_year, Float32(row.trim_rank))
            vk_to_group[vk] = (row.model, row.model_year, row.extra)
        end

        intercept_col(j_idx) = j_idx
        coefficient_col(j_idx, k_idx) = j_idx + k_idx * J
        tlag_of(t) = t[2] == 1 ? (t[1] - 1, 4) : (t[1], t[2] - 1)

        function panel_for_product(j_idx)
            vk = kept_vks[j_idx]
            rows_t = Int[]
            rows_tlag = Int[]
            carried = Vector{Vector{Int}}()
            for (tf, i_t) in state_lookup
                (t, f) = tf
                js_at_t = intermed[tf]
                vk in js_at_t || continue
                tlag = tlag_of(t)
                haskey(state_lookup, (tlag, f)) || continue
                push!(rows_t, i_t)
                push!(rows_tlag, state_lookup[(tlag, f)])
                push!(carried, [prod_index[k] for k in js_at_t])
            end
            return rows_t, rows_tlag, carried
        end

        function bootup(bytes, ncols)
            isempty(bytes) && return nothing
            dummy = XGBoost.DMatrix(zeros(Float32, 1, ncols))
            b = XGBoost.Booster(dummy)
            XGBoost.load!(b, bytes)
            finalize(dummy)
            return b
        end

        AccKey = Tuple{String, String, Int64, String, String, Float32, Int, Int}
        acc = Dict{AccKey, Vector{Float64}}()
        function record!(fam, grp, target, τ, year, quarter, hit, resid)
            k = (fam, grp[1], grp[2], grp[3], target, τ, year, quarter)
            v = get!(acc, k, zeros(Float64, 4))
            v[1] += hit
            v[2] += resid
            v[3] += abs(resid)
            v[4] += 1
            return nothing
        end

        CovKey = Tuple{String, String, Int64, String, Int, Int}
        cov = Dict{CovKey, Int}()
        function record_cov!(fam, grp, year, quarter)
            k = (fam, grp[1], grp[2], grp[3], year, quarter)
            cov[k] = get(cov, k, 0) + 1
            return nothing
        end

        state_transition = JLD2.load_object(snakemake.input["state_transition"])
        price_densities = JLD2.load_object(snakemake.input["price_densities"])

        n_vks_st = 0
        for (j_idx, vk) in enumerate(kept_vks)
            haskey(state_transition, vk) || continue
            haskey(vk_to_group, vk) || continue
            grp = vk_to_group[vk]
            int_bytes, own_bytes, cross_bytes = state_transition[vk]

            rows_t, rows_tlag, carried = panel_for_product(j_idx)
            n = length(rows_t)
            n == 0 && continue

            X_common = Matrix{Float32}(undef, n, ncols_state + 2)
            for r in 1:n
                X_common[r, 1:ncols_state] .= @view state_matrix[rows_tlag[r], :]
                t_r = tf_keys[rows_t[r]][1]
                X_common[r, ncols_state + 1] = Float32(log(Float64(J_by_t[t_r])))
                X_common[r, ncols_state + 2] = Float32(t_r[2])
            end
            y_int = Float32[state_matrix[rows_t[r], intercept_col(j_idx)] for r in 1:n]
            y_own = Float32[state_matrix[rows_t[r], coefficient_col(j_idx, j_idx)] for r in 1:n]
            years = Int[tf_keys[rows_t[r]][1][1] for r in 1:n]
            quarters = Int[tf_keys[rows_t[r]][1][2] for r in 1:n]

            for r in 1:n
                record_cov!("state_transition", grp, years[r], quarters[r])
            end

            dm = XGBoost.DMatrix(X_common)
            for (ty, byte_grid, y) in (
                ("intercept", int_bytes, y_int),
                ("own", own_bytes, y_own),
            )
                for (qi, τ) in enumerate(τs)
                    b = bootup(byte_grid[qi], ncols_state + 2)
                    b === nothing && continue
                    ŷ = XGBoost.predict(b, dm; margin = true)
                    finalize(b)
                    for r in 1:n
                        hit = Float64(y[r] ≤ ŷ[r])
                        resid = Float64(y[r] - ŷ[r])
                        record!(
                            "state_transition", grp, ty, τ, years[r], quarters[r], hit, resid
                        )
                    end
                end
            end
            finalize(dm)

            cross_r = Int[]
            cross_k = Int[]
            cross_y = Float32[]
            for r in 1:n
                for k_idx in carried[r]
                    k_idx == j_idx && continue
                    push!(cross_r, r)
                    push!(cross_k, k_idx)
                    push!(cross_y, state_matrix[rows_t[r], coefficient_col(j_idx, k_idx)])
                end
            end
            n_cross = length(cross_r)
            if n_cross > 0 && !all(isempty, cross_bytes)
                X_cross = Matrix{Float32}(undef, n_cross, ncols_state + 3)
                for i in 1:n_cross
                    X_cross[i, 1:(ncols_state + 2)] .= @view X_common[cross_r[i], :]
                    X_cross[i, end] = Float32(cross_k[i])
                end
                dm_cross = XGBoost.DMatrix(X_cross)
                for (qi, τ) in enumerate(τs)
                    b = bootup(cross_bytes[qi], ncols_state + 3)
                    b === nothing && continue
                    ŷ = XGBoost.predict(b, dm_cross; margin = true)
                    finalize(b)
                    for i in 1:n_cross
                        hit = Float64(cross_y[i] ≤ ŷ[i])
                        resid = Float64(cross_y[i] - ŷ[i])
                        yr = years[cross_r[i]]
                        qr = quarters[cross_r[i]]
                        record!("state_transition", grp, "cross", τ, yr, qr, hit, resid)
                    end
                end
                finalize(dm_cross)
            end

            n_vks_st += 1
        end
        @info "State-transition calibration" n_vks = n_vks_st

        _sales_bytes, _τ_sales = JLD2.load_object(snakemake.input["sales_targets"])
        compliance_arrow = read_arrow(snakemake.input["compliance"])
        ι_lag = Dict{Tuple{QuarterKey, UUID}, Float32}()
        for r in eachrow(compliance_arrow)
            t_next = r.t + Quarter(1)
            key = ((Dates.year(t_next), Dates.quarterofyear(t_next)), r.dealer_uid)
            ι_lag[key] = Float32(r.p_compliant > 0.5)
        end

        eligible = Tuple{Tuple{QuarterKey, UUID}, Int}[]
        for (tf, _) in state_lookup
            haskey(ι_lag, tf) || continue
            tlag = tlag_of(tf[1])
            haskey(state_lookup, (tlag, tf[2])) || continue
            push!(eligible, (tf, state_lookup[(tlag, tf[2])]))
        end

        logphi_lookup = Dict{Tuple{QuarterKey, UUID}, Float32}()
        if !isempty(eligible)
            n_phi = length(eligible)
            X_phi = Matrix{Float32}(undef, n_phi, ncols_state + 2)
            for r in 1:n_phi
                tf, i_tlag = eligible[r]
                X_phi[r, 1:ncols_state] .= @view state_matrix[i_tlag, :]
                X_phi[r, ncols_state + 1] = Float32(tf[1][2])
                X_phi[r, end] = ι_lag[tf]
            end
            b_sales = bootup(_sales_bytes, ncols_state + 2)
            dm_phi = XGBoost.DMatrix(X_phi)
            logphi_vals = XGBoost.predict(b_sales, dm_phi; margin = true)
            finalize(dm_phi)
            finalize(b_sales)
            for r in 1:n_phi
                tf, _ = eligible[r]
                logphi_lookup[tf] = Float32(logphi_vals[r])
            end
        end

        base = Supply.tf_to_j_lookup(model)
        idx_dict = Dict(k => i for (i, k) in enumerate(tf_keys))
        tjf_dict = model.product_lookup.tjf
        j_to_tf = Dict{VehicleKey, Vector{Tuple{Tuple{QuarterKey, UUID}, Int64}}}()
        for (tf, js) in base
            haskey(idx_dict, tf) || continue
            haskey(logphi_lookup, tf) || continue
            for j in js
                j ∈ kept_set || continue
                push!(
                    get!(j_to_tf, j, Tuple{Tuple{QuarterKey, UUID}, Int64}[]),
                    (tf, idx_dict[tf]),
                )
            end
        end

        n_vks_pd = 0
        for (vk, byte_grid) in price_densities
            haskey(j_to_tf, vk) || continue
            haskey(vk_to_group, vk) || continue
            grp = vk_to_group[vk]
            rows = j_to_tf[vk]
            n_r = length(rows)
            n_r == 0 && continue
            X_j = Matrix{Float32}(undef, n_r, ncols_state + 1)
            y_j = Vector{Float32}(undef, n_r)
            yqs = Vector{Tuple{Int, Int}}(undef, n_r)
            for (ri, (tf, i_row)) in enumerate(rows)
                X_j[ri, 1:ncols_state] .= @view state_matrix[i_row, :]
                X_j[ri, end] = logphi_lookup[tf]
                y_j[ri] = Float32(log(model.product_info[tjf_dict[(tf[1], vk, tf[2])]].p))
                yqs[ri] = (tf[1][1], tf[1][2])
                record_cov!("price_density", grp, yqs[ri][1], yqs[ri][2])
            end

            dm_j = XGBoost.DMatrix(X_j)
            for (qi, τ) in enumerate(τs)
                b = bootup(byte_grid[qi], ncols_state + 1)
                b === nothing && continue
                ŷ = XGBoost.predict(b, dm_j; margin = true)
                finalize(b)
                for ri in 1:n_r
                    hit = Float64(y_j[ri] ≤ ŷ[ri])
                    resid = Float64(y_j[ri] - ŷ[ri])
                    year, quarter = yqs[ri]
                    record!("price_density", grp, "log_price", τ, year, quarter, hit, resid)
                end
            end
            finalize(dm_j)
            n_vks_pd += 1
        end
        @info "Price-density calibration" n_vks = n_vks_pd

        cal_rows = DataFrame(
            family = String[], group_model = String[], group_my = Int64[],
            group_extra = String[],
            target_type = String[], tau = Float32[], year = Int[], quarter = Int[],
            n_rows = Int[], hit_rate = Float64[], cal_err = Float64[],
            mean_resid = Float64[], mean_abs_resid = Float64[],
        )
        for (k, v) in acc
            fam, m, my, extra, target, τ, yr, qr = k
            n = v[4]
            n > 0 || continue
            hr = v[1] / n
            push!(
                cal_rows,
                (
                    fam, m, my, extra, target, τ, yr, qr,
                    round(Int, n), hr, hr - Float64(τ), v[2] / n, v[3] / n,
                ),
            )
        end
        sort!(
            cal_rows,
            [:family, :target_type, :group_model, :group_my, :group_extra, :tau, :year, :quarter],
        )
        Arrow.write(snakemake.output["calibration"], cal_rows)
        @info "Calibration table" n_rows = nrow(cal_rows)

        cov_rows = DataFrame(
            family = String[], group_model = String[], group_my = Int64[],
            group_extra = String[],
            year = Int[], quarter = Int[], n_train = Int[],
        )
        for (k, v) in cov
            fam, m, my, extra, yr, qr = k
            push!(cov_rows, (fam, m, my, extra, yr, qr, v))
        end
        sort!(
            cov_rows,
            [:family, :group_model, :group_my, :group_extra, :year, :quarter],
        )
        Arrow.write(snakemake.output["coverage"], cov_rows)
        @info "Coverage table" n_rows = nrow(cov_rows)
    end
    return nothing
end
