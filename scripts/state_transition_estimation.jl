using Distributed

@everywhere using Memoize

@everywhere begin
    using DealerMnA
    using .Demand
    using XGBoost
    using LoggingExtras
    using LinearAlgebra
    using Statistics: mean
    using Random
    import Base: size  # disambiguate against Cleaner.size (re-exported via DealerMnA)

    BLAS.set_num_threads(1)

    model = Demand.DemandModel(; JLD2.load_object("data/model_frames.jld2")...)
    state_matrix = JLD2.load_object("data/residual_demand.jld2")

    intermed = Supply.tf_to_j_lookup(model)
    state_lookup = Dict(k => i for (i, k) in enumerate(keys(intermed)))
    tf_keys = collect(keys(intermed))

    products = collect(keys(model.product_lookup.j))
    J = length(products)
    prod_index = Dict(p => i for (i, p) in enumerate(products))

    # Kept-trim filter: fit boosters only for VehicleKeys whose (model, extra)
    # cleared the share cutoff, and only train cross-price predictions on
    # partners that also cleared it. Same filter as price_density_estimation
    # and residual_demands so BBL sees a consistent VK universe.
    _kept_pairs = Set(
        (row.model, row.extra) for
        row in eachrow(read_arrow("data/kept_trims.arrow"))
    )
    _price_table = read_arrow("data/price_table.arrow")
    _vk_to_extra = Dict{VehicleKey, String}()
    for row in eachrow(_price_table)
        _vk_to_extra[(row.make, row.model, row.model_year, Float32(row.trim_rank))] =
            row.extra
    end
    kept_vks = Set(
        j for j in keys(model.product_lookup.j) if
        haskey(_vk_to_extra, j) && (j[2], _vk_to_extra[j]) ∈ _kept_pairs
    )
    kept_j_idxs = sort([prod_index[j] for j in kept_vks])

    # Column-major flat layout for the J × (J+1) residual_demand matrix per row:
    #   intercept for product j: column j
    #   coefficient with product k: column j + k*J
    # (k = j gives the own-price coefficient.)
    intercept_col(j_idx) = j_idx
    coefficient_col(j_idx, k_idx) = j_idx + k_idx * J

    tlag_of(t) = t[2] == 1 ? (t[1] - 1, 4) : (t[1], t[2] - 1)

    # Panel: (t, f) rows where j is carried at t and (t-1, f) is in state_lookup.
    function panel_for_product(j_idx::Int)
        vk = products[j_idx]
        rows_t = Int[]
        rows_tlag = Int[]
        carried = Vector{Vector{Int}}()
        for ((t, f), i_t) in state_lookup
            js_at_t = intermed[(t, f)]
            vk in js_at_t || continue
            tlag = tlag_of(t)
            haskey(state_lookup, (tlag, f)) || continue
            push!(rows_t, i_t)
            push!(rows_tlag, state_lookup[(tlag, f)])
            # Restrict cross-price partners to kept VKs — BBL only ever
            # samples kept assortments, so unfiltered partners would train
            # on rows the sim can't produce.
            push!(carried, [prod_index[k] for k in js_at_t if k ∈ kept_vks])
        end
        return rows_t, rows_tlag, carried
    end

    function held_out_r2(booster, X_val, y_val)
        ŷ = XGBoost.predict(booster, X_val)
        ss_res = sum((y_val .- ŷ) .^ 2)
        ss_tot = sum((y_val .- mean(y_val)) .^ 2)
        return ss_tot > 0 ? 1 - ss_res / ss_tot : NaN
    end

    function fit_product(j_idx::Int)
        rows_t, rows_tlag, carried = panel_for_product(j_idx)
        n = length(rows_t)
        if n < 20
            @info "Skipping product $j_idx (n=$n rows)"
            return nothing
        end

        ncols_state = size(state_matrix, 2)
        X_common = Matrix{Float32}(undef, n, ncols_state)
        for r in 1:n
            X_common[r, :] .= @view state_matrix[rows_tlag[r], :]
        end

        y_int = Float32[state_matrix[rows_t[r], intercept_col(j_idx)] for r in 1:n]
        y_own = Float32[state_matrix[rows_t[r], coefficient_col(j_idx, j_idx)] for r in 1:n]

        # Cross-price training rows: one per (panel row, partner k) with k ≠ j.
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
        X_cross = Matrix{Float32}(undef, n_cross, ncols_state + 1)
        for i in 1:n_cross
            X_cross[i, 1:ncols_state] .= @view X_common[cross_r[i], :]
            X_cross[i, end] = Float32(cross_k[i])
        end

        rng = MersenneTwister(hash(products[j_idx]) & 0xffffffff)

        params_shared = (subsample = 0.8,)
        fit_scalar(X, y) = Supply.fit_xgb_oos(
            X, y;
            max_round = 500, patience = 5,
            params = merge(params_shared, (base_score = Float64(mean(y)),)),
            rng = rng,
        )

        # For each of {intercept, own, cross} we fit a mean booster and a
        # variance booster (squared-residual target). BBL draws the transition
        # as Normal(mean, sqrt(clip(var, 0))). Same 2-step density pattern as
        # price_density_estimation.
        function fit_pair(X, y)
            b_mean, br_mean = fit_scalar(X, y)
            ŷ = XGBoost.predict(b_mean, X)
            u = (y .- ŷ) .^ 2
            b_var, br_var = fit_scalar(X, u)
            return b_mean, br_mean, b_var, br_var
        end

        b_int_mean, br_int_mean, b_int_var, br_int_var = fit_pair(X_common, y_int)
        b_own_mean, br_own_mean, b_own_var, br_own_var = fit_pair(X_common, y_own)

        b_cross_mean, br_cross_mean, b_cross_var, br_cross_var = if n_cross >= 20
            fit_pair(X_cross, cross_y)
        else
            nothing, 0, nothing, 0
        end

        # Held-out R² on a fresh 20% split (cheap diagnostic; not the OOS split
        # that fit_xgb_oos uses internally). Only on the mean boosters — the
        # variance boosters don't have a natural R² baseline.
        function r2_diag(X, y, b)
            b === nothing && return NaN
            perm = randperm(rng, length(y))
            n_val = max(1, floor(Int, 0.2 * length(y)))
            val_idx = perm[1:n_val]
            dm_val = XGBoost.DMatrix(X[val_idx, :])
            r2 = held_out_r2(b, dm_val, y[val_idx])
            finalize(dm_val)
            return r2
        end
        r2_int = r2_diag(X_common, y_int, b_int_mean)
        r2_own = r2_diag(X_common, y_own, b_own_mean)
        r2_cross = r2_diag(X_cross, cross_y, b_cross_mean)

        @info "product fit" j_idx n n_cross br_int_mean br_int_var br_own_mean br_own_var br_cross_mean br_cross_var r2_int r2_own r2_cross

        # `XGBoost.save` returns an unsafe_wrap over libxgboost-owned memory;
        # copy immediately so the bytes survive booster finalization / GC.
        save_ubj(b) = copy(XGBoost.save(b, Vector{UInt8}; format = "ubj"))
        bytes_int_mean = save_ubj(b_int_mean)
        bytes_int_var = save_ubj(b_int_var)
        bytes_own_mean = save_ubj(b_own_mean)
        bytes_own_var = save_ubj(b_own_var)
        bytes_cross_mean = b_cross_mean === nothing ? UInt8[] : save_ubj(b_cross_mean)
        bytes_cross_var = b_cross_var === nothing ? UInt8[] : save_ubj(b_cross_var)
        finalize(b_int_mean)
        finalize(b_int_var)
        finalize(b_own_mean)
        finalize(b_own_var)
        b_cross_mean === nothing || finalize(b_cross_mean)
        b_cross_var === nothing || finalize(b_cross_var)

        return (
            bytes_int_mean, bytes_int_var,
            bytes_own_mean, bytes_own_var,
            bytes_cross_mean, bytes_cross_var,
        )
    end
end

function (@main)(_)
    @info "Fitting state transitions" n_kept = length(kept_j_idxs) n_dropped = length(
        products
    ) - length(kept_j_idxs)
    fitted = pmap(kept_j_idxs) do j_idx
        try
            fit_product(j_idx)
        catch e
            @error "Product $j_idx failed" j_idx exception = (e, catch_backtrace())
            return nothing
        end
    end
    out = Dict{
        VehicleKey,
        Tuple{
            Vector{UInt8}, Vector{UInt8},  # intercept: mean, var
            Vector{UInt8}, Vector{UInt8},  # own:        mean, var
            Vector{UInt8}, Vector{UInt8},  # cross:      mean, var
        },
    }()
    for (j_idx, f) in zip(kept_j_idxs, fitted)
        f === nothing && continue
        out[products[j_idx]] = f
    end
    JLD2.save_object("data/state_transition.jld2", out)
    return nothing
end
