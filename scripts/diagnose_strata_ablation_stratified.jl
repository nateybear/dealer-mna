using Distributed

@everywhere begin
    using DealerMnA
    using .Demand
    using XGBoost
    using DataFrames
    using Statistics: mean
    using Random
    using Dates
    import Base: size

    SYMPTOMATIC_GROUPS = [
        ("Escape", 2018, "SE"),
        ("Escape", 2018, "S"),
        ("Fusion", 2018, "FWD SE"),
        ("Explorer", 2018, "XLT"),
        ("Explorer", 2018, "LIMITED"),
        ("Mustang", 2018, "COUPE I4"),
        ("Mustang", 2018, "COUPE GT"),
        ("Escape", 2018, "SEL"),
        ("Edge", 2018, "SEL"),
        ("Focus", 2018, "SE"),
    ]
    const SPLIT_SEED = UInt64(20261103)
    const FIT_SEED = UInt64(20261104)

    model = DemandModel(; JLD2.load_object("data/model_frames.jld2")...)
    _rd = JLD2.load_object("data/residual_demand.jld2")
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

    intercept_col(j_idx) = j_idx
    coefficient_col(j_idx, k_idx) = j_idx + k_idx * J
    tlag_of(t) = t[2] == 1 ? (t[1] - 1, 4) : (t[1], t[2] - 1)

    _sales_bytes, _τ_sales = JLD2.load_object("data/sales_targets_stratified.jld2")
    compliance_arrow = read_arrow("data/compliance_stratified.arrow")
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
        X_phi_mat = Matrix{Float32}(undef, n_phi, ncols_state + 2)
        for r in 1:n_phi
            tf, i_tlag = eligible[r]
            X_phi_mat[r, 1:ncols_state] .= @view state_matrix[i_tlag, :]
            X_phi_mat[r, ncols_state + 1] = Float32(tf[1][2])
            X_phi_mat[r, end] = ι_lag[tf]
        end
        let dummy = XGBoost.DMatrix(zeros(Float32, 1, ncols_state + 2))
            b_sales = XGBoost.Booster(dummy)
            XGBoost.load!(b_sales, _sales_bytes)
            finalize(dummy)
            dm_phi = XGBoost.DMatrix(X_phi_mat)
            logphi_vals = XGBoost.predict(b_sales, dm_phi; margin = true)
            finalize(dm_phi)
            finalize(b_sales)
            for r in 1:n_phi
                tf, _ = eligible[r]
                logphi_lookup[tf] = Float32(logphi_vals[r])
            end
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

    params_shared = merge(
        Supply.XGB_DEFAULT_PARAMS,
        (subsample = 0.8, max_depth = 8, min_child_weight = 5, eta = 0.05, lambda = 1.0),
    )

    function panel_for_product(j_idx)
        vk = kept_vks[j_idx]
        rows_t = Int[]
        rows_tlag = Int[]
        for (tf, i_t) in state_lookup
            vk in intermed[tf] || continue
            tlag = tlag_of(tf[1])
            haskey(state_lookup, (tlag, tf[2])) || continue
            push!(rows_t, i_t)
            push!(rows_tlag, state_lookup[(tlag, tf[2])])
        end
        return rows_t, rows_tlag
    end

    function build_X_y(vk, target)
        j_idx = get(prod_index, vk, 0)
        j_idx == 0 && return nothing
        if target == "intercept" || target == "own"
            rows_t, rows_tlag = panel_for_product(j_idx)
            n = length(rows_t)
            n == 0 && return nothing
            X = Matrix{Float32}(undef, n, ncols_state)
            for r in 1:n
                X[r, :] .= @view state_matrix[rows_tlag[r], :]
            end
            col =
                target == "intercept" ? intercept_col(j_idx) :
                coefficient_col(j_idx, j_idx)
            y = Float32[state_matrix[rows_t[r], col] for r in 1:n]
            yqs = Tuple{Int, Int}[
                (tf_keys[rows_t[r]][1][1], tf_keys[rows_t[r]][1][2]) for r in 1:n
            ]
            return X, y, yqs
        elseif target == "log_price"
            haskey(j_to_tf, vk) || return nothing
            rows = j_to_tf[vk]
            n = length(rows)
            n == 0 && return nothing
            X = Matrix{Float32}(undef, n, ncols_state + 1)
            y = Vector{Float32}(undef, n)
            yqs = Vector{Tuple{Int, Int}}(undef, n)
            for (ri, (tf, i_row)) in enumerate(rows)
                X[ri, 1:ncols_state] .= @view state_matrix[i_row, :]
                X[ri, end] = logphi_lookup[tf]
                y[ri] = Float32(log(model.product_info[tjf_dict[(tf[1], vk, tf[2])]].p))
                yqs[ri] = (tf[1][1], tf[1][2])
            end
            return X, y, yqs
        else
            error("unknown target: $target")
        end
    end

    function pinball(y, ŷ, τ)
        s = 0.0
        for i in eachindex(y)
            d = Float64(y[i] - ŷ[i])
            s += d ≥ 0 ? τ * d : (τ - 1.0) * d
        end
        return s / length(y)
    end

    function run_vk_target(vk, target)
        xy = build_X_y(vk, target)
        xy === nothing && return DataFrame()
        X_all, y_all, yqs_all = xy
        n_all = length(y_all)
        n_all < 50 && return DataFrame()

        rng_split = MersenneTwister(SPLIT_SEED ⊻ (UInt64(hash(vk)) & 0xffffffff))
        tr_idx, ho_idx = Supply._stratified_split(yqs_all, 0.2, rng_split)
        isempty(ho_idx) && return DataFrame()
        X_tr = X_all[tr_idx, :]
        y_tr = y_all[tr_idx]
        yqs_tr = yqs_all[tr_idx]
        X_ho = X_all[ho_idx, :]
        y_ho = y_all[ho_idx]

        out = DataFrame(
            vk_model = String[], vk_my = Int[], vk_extra_rank = Float32[],
            target = String[], tau = Float32[], arm = String[],
            n_train = Int[], n_holdout = Int[],
            hit_rate = Float64[], cal_err = Float64[],
            median_resid = Float64[], pinball_loss = Float64[],
        )
        ncols = size(X_tr, 2)
        dm_ho = XGBoost.DMatrix(X_ho)
        for arm_label in ("nostrata", "qstrata")
            seed_mask = UInt64(hash((vk, target, arm_label))) & 0xffffffff
            rng_fit = MersenneTwister(FIT_SEED ⊻ seed_mask)
            strata = arm_label == "qstrata" ? yqs_tr : nothing
            byte_grid = Supply.fit_quantile_grid(
                X_tr, y_tr;
                max_round = 2000, patience = 20,
                params = params_shared,
                strata = strata,
                rng = rng_fit,
            )
            for (qi, τ) in enumerate(τs)
                isempty(byte_grid[qi]) && continue
                dummy = XGBoost.DMatrix(zeros(Float32, 1, ncols))
                b = XGBoost.Booster(dummy)
                XGBoost.load!(b, byte_grid[qi])
                finalize(dummy)
                ŷ = XGBoost.predict(b, dm_ho; margin = true)
                finalize(b)
                hr = Float64(mean(y_ho .≤ ŷ))
                mresid = τ ≈ 0.5f0 ? Float64(mean(y_ho .- ŷ)) : NaN
                pb = pinball(y_ho, ŷ, Float64(τ))
                push!(
                    out,
                    (
                        vk[2], vk[3], vk[4], target, τ, arm_label,
                        length(y_tr), length(y_ho),
                        hr, hr - Float64(τ), mresid, pb,
                    ),
                )
            end
        end
        finalize(dm_ho)
        return out
    end
end

function (@main)(_)
    snakemake!()
    with_logging() do
        price_table = read_arrow(snakemake.input["price_table"])
        vk_by_group = Dict{Tuple{String, Int64, String}, VehicleKey}()
        for row in eachrow(price_table)
            vk = (row.make, row.model, row.model_year, Float32(row.trim_rank))
            vk ∈ kept_set || continue
            vk_by_group[(row.model, row.model_year, row.extra)] = vk
        end
        vks = VehicleKey[]
        for g in SYMPTOMATIC_GROUPS
            if haskey(vk_by_group, g)
                push!(vks, vk_by_group[g])
            else
                @warn "Group not in kept_vks" g
            end
        end
        @info "Symptomatic VKs" n = length(vks)

        tasks = [(vk, target) for vk in vks for target in ("intercept", "own", "log_price")]
        results = pmap(tasks) do (vk, target)
            try
                run_vk_target(vk, target)
            catch e
                @error "VK/target failed" vk target exception = (e, catch_backtrace())
                return DataFrame()
            end
        end
        all_rows = reduce(vcat, results; init = DataFrame())
        if nrow(all_rows) > 0
            sort!(all_rows, [:target, :vk_model, :vk_my, :vk_extra_rank, :arm, :tau])
        end
        Arrow.write(snakemake.output["ablation"], all_rows)
        @info "Ablation table" n_rows = nrow(all_rows)
    end
    return nothing
end
