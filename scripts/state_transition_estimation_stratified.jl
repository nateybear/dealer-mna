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

        params_shared = merge(
            Supply.XGB_DEFAULT_PARAMS,
            (
                subsample = 0.8,
                max_depth = 8,
                min_child_weight = 5,
                eta = 0.05,
                lambda = 1.0,
            ),
        )

        # Nonparametric conditional density via quantile-regression grid: one
        # XGBoost booster per level in Supply.QUANTILE_LEVELS, per target
        # (intercept, own, cross). BBL samples the transition by inverse-CDF
        # with linear interpolation between adjacent quantile knots. Replaces
        # the 2-step mean+variance Normal-draw pattern.
        fit_grid(X, y) = Supply.fit_quantile_grid(
            X, y;
            max_round = 2000, patience = 20,
            params = params_shared,
            rng = rng,
        )

        int_bytes = fit_grid(X_common, y_int)
        own_bytes = fit_grid(X_common, y_own)
        cross_bytes = n_cross >= 20 ? fit_grid(X_cross, cross_y) :
                      [UInt8[] for _ in Supply.QUANTILE_LEVELS]

        # Held-out R² on the median booster only — cheap sanity check that the
        # signal is nontrivial. Uses a fresh 20% split (not the OOS split
        # fit_quantile_grid uses internally). The 0.5 booster's median
        # prediction is the closest quantile-regression analogue to the old
        # mean fit's ŷ.
        function r2_diag_median(X, y, grid_bytes)
            all(isempty, grid_bytes) && return NaN
            median_idx = findfirst(≈(0.5f0), Supply.QUANTILE_LEVELS)
            median_bytes = grid_bytes[median_idx]
            isempty(median_bytes) && return NaN
            perm = randperm(rng, length(y))
            n_val = max(1, floor(Int, 0.2 * length(y)))
            val_idx = perm[1:n_val]
            dm_val = XGBoost.DMatrix(X[val_idx, :])
            booster = XGBoost.Booster(dm_val)
            XGBoost.load!(booster, median_bytes)
            r2 = held_out_r2(booster, dm_val, y[val_idx])
            finalize(booster)
            finalize(dm_val)
            return r2
        end
        r2_int = r2_diag_median(X_common, y_int, int_bytes)
        r2_own = r2_diag_median(X_common, y_own, own_bytes)
        r2_cross = r2_diag_median(X_cross, cross_y, cross_bytes)

        n_active_int = count(!isempty, int_bytes)
        n_active_own = count(!isempty, own_bytes)
        n_active_cross = count(!isempty, cross_bytes)
        @info "product fit" j_idx n n_cross n_active_int n_active_own n_active_cross r2_int r2_own r2_cross

        return (int_bytes, own_bytes, cross_bytes)
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
    # Layout per VK: (int_quantiles, own_quantiles, cross_quantiles), each a
    # Vector{Vector{UInt8}} with one saved booster per level in
    # Supply.QUANTILE_LEVELS. cross_quantiles may contain all-empty entries
    # (n_cross < 20) — BBL sim treats that as "no cross partners for this VK".
    out = Dict{
        VehicleKey,
        Tuple{
            Vector{Vector{UInt8}},  # intercept
            Vector{Vector{UInt8}},  # own
            Vector{Vector{UInt8}},  # cross
        },
    }()
    for (j_idx, f) in zip(kept_j_idxs, fitted)
        f === nothing && continue
        out[products[j_idx]] = f
    end
    JLD2.save_object("data/state_transition_stratified.jld2", out)
    return nothing
end
