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

        b_int, br_int = fit_scalar(X_common, y_int)
        b_own, br_own = fit_scalar(X_common, y_own)

        b_cross, br_cross = if n_cross >= 20
            fit_scalar(X_cross, cross_y)
        else
            nothing, 0
        end

        # Held-out R² on a fresh 20% split (cheap diagnostic; not the OOS split
        # that fit_xgb_oos uses internally).
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
        r2_int = r2_diag(X_common, y_int, b_int)
        r2_own = r2_diag(X_common, y_own, b_own)
        r2_cross = r2_diag(X_cross, cross_y, b_cross)

        @info "product fit" j_idx n n_cross br_int br_own br_cross r2_int r2_own r2_cross

        bytes_int = XGBoost.serialize(b_int)
        bytes_own = XGBoost.serialize(b_own)
        bytes_cross = b_cross === nothing ? UInt8[] : XGBoost.serialize(b_cross)
        finalize(b_int)
        finalize(b_own)
        b_cross === nothing || finalize(b_cross)

        return (bytes_int, bytes_own, bytes_cross)
    end
end

function (@main)(_)
    fitted = pmap(1:length(products)) do j_idx
        try
            fit_product(j_idx)
        catch e
            @error "Product $j_idx failed" j_idx exception = (e, catch_backtrace())
            return nothing
        end
    end
    out = Dict{VehicleKey, Tuple{Vector{UInt8}, Vector{UInt8}, Vector{UInt8}}}()
    for (j_idx, f) in enumerate(fitted)
        f === nothing && continue
        out[products[j_idx]] = f
    end
    JLD2.save_object("data/state_transition.jld2", out)
    return nothing
end
