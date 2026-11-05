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
    import Base: size

    BLAS.set_num_threads(1)

    model = Demand.DemandModel(; JLD2.load_object("data/model_frames.jld2")...)
    _rd = JLD2.load_object("data/residual_demand.jld2")
    state_matrix = _rd.matrix
    products = _rd.kept_vks
    J = length(products)
    prod_index = Dict(p => i for (i, p) in enumerate(products))
    kept_set = Set(products)

    state_lookup = Dict(k => i for (i, k) in enumerate(_rd.tf_keys))
    tf_keys = _rd.tf_keys
    intermed_raw = Supply.tf_to_j_lookup(model)
    intermed = Dict(
        k => VehicleKey[j for j in get(intermed_raw, k, VehicleKey[]) if j ∈ kept_set]
        for k in tf_keys
    )

    kept_j_idxs = collect(1:J)

    # Flat state_matrix col for focal j, by layout slot. Row within the K×(2K+1)
    # block is j (column-major reshape), col is Supply.intercept_col() etc.
    @inline flat_intercept(j) = j
    @inline flat_firstorder(j, k) = k * J + j
    @inline flat_quadratic(j, k) = (J + k) * J + j

    tlag_of(t) = t[2] == 1 ? (t[1] - 1, 4) : (t[1], t[2] - 1)

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
            push!(carried, [prod_index[k] for k in js_at_t])
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

        y_int = Float32[state_matrix[rows_t[r], flat_intercept(j_idx)] for r in 1:n]
        y_own_fo = Float32[state_matrix[rows_t[r], flat_firstorder(j_idx, j_idx)] for r in 1:n]
        y_own_so = Float32[state_matrix[rows_t[r], flat_quadratic(j_idx, j_idx)] for r in 1:n]

        cross_r = Int[]
        cross_k = Int[]
        cross_y_fo = Float32[]
        cross_y_int = Float32[]
        for r in 1:n
            for k_idx in carried[r]
                k_idx == j_idx && continue
                push!(cross_r, r)
                push!(cross_k, k_idx)
                push!(cross_y_fo, state_matrix[rows_t[r], flat_firstorder(j_idx, k_idx)])
                push!(cross_y_int, state_matrix[rows_t[r], flat_quadratic(j_idx, k_idx)])
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

        fit_grid(X, y; strata = nothing) = Supply.fit_quantile_grid(
            X, y;
            max_round = 2000, patience = 20,
            params = params_shared,
            strata = strata,
            rng = rng,
        )

        empty_grid() = [UInt8[] for _ in Supply.QUANTILE_LEVELS]

        int_bytes = fit_grid(X_common, y_int)
        own_fo_bytes = fit_grid(X_common, y_own_fo)
        own_so_bytes = fit_grid(X_common, y_own_so)
        cross_fo_bytes = n_cross >= 20 ?
            fit_grid(X_cross, cross_y_fo; strata = cross_k) : empty_grid()
        cross_int_bytes = n_cross >= 20 ?
            fit_grid(X_cross, cross_y_int; strata = cross_k) : empty_grid()

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
        r2_own_fo = r2_diag_median(X_common, y_own_fo, own_fo_bytes)
        r2_own_so = r2_diag_median(X_common, y_own_so, own_so_bytes)
        r2_cross_fo = r2_diag_median(X_cross, cross_y_fo, cross_fo_bytes)
        r2_cross_int = r2_diag_median(X_cross, cross_y_int, cross_int_bytes)

        @info "product fit" j_idx n n_cross r2_int r2_own_fo r2_own_so r2_cross_fo r2_cross_int

        return (;
            intercept = int_bytes,
            own_firstorder = own_fo_bytes,
            own_secondorder = own_so_bytes,
            cross_firstorder = cross_fo_bytes,
            cross_interaction = cross_int_bytes,
        )
    end
end

function (@main)(_)
    @info "Fitting state transitions" n_kept = length(kept_j_idxs)
    fitted = pmap(kept_j_idxs) do j_idx
        try
            fit_product(j_idx)
        catch e
            @error "Product $j_idx failed" j_idx exception = (e, catch_backtrace())
            return nothing
        end
    end
    out = Dict{VehicleKey, NamedTuple}()
    for (j_idx, f) in zip(kept_j_idxs, fitted)
        f === nothing && continue
        out[products[j_idx]] = f
    end
    JLD2.save_object("data/state_transition.jld2", out)
    return nothing
end
