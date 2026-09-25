using Distributed

@everywhere using Memoize

@everywhere begin
    using DealerMnA
    using .Demand
    using XGBoost
    using LoggingExtras
    using Statistics: mean
    using Dates
    import Base: size  # disambiguate against Cleaner.size (re-exported via DealerMnA)

    model = DemandModel(; JLD2.load_object("data/model_frames.jld2")...)
    _rd = JLD2.load_object("data/residual_demand.jld2")
    state_matrix = _rd.matrix

    # Sales-target booster + τ (from data_sales_target_estimation) — used to
    # compute log φ_{t,f} as an additional conditioning feature for prices.
    _sales_bytes, τ_sales = JLD2.load_object("data/sales_targets_stratified.jld2")

    # Precompute log φ_{t,f} for every (t, f) where the sales-target booster's
    # inputs are available (state_{t−1, f} in state_matrix and p_compliant at
    # t−1 in compliance.arrow). Rows without a lagged compliance entry drop
    # out — dealer's first observed quarter.
    @memoize function logphi_lookup()
        state_lookup = Dict(k => i for (i, k) in enumerate(_rd.tf_keys))
        tlag_of(t) = t[2] == 1 ? (t[1] - 1, 4) : (t[1], t[2] - 1)

        # Sales-target booster was trained with ι_lag as a hard 1{p_compliant > 0.5}
        # indicator — use the same threshold here so the input matches training.
        compliance = read_arrow("data/compliance_stratified.arrow")
        ι_lag = Dict{Tuple{QuarterKey, UUID}, Float32}()
        for r in eachrow(compliance)
            t_next = r.t + Quarter(1)
            key = ((Dates.year(t_next), Dates.quarterofyear(t_next)), r.dealer_uid)
            ι_lag[key] = Float32(r.p_compliant > 0.5)
        end

        eligible = Tuple{Tuple{QuarterKey, UUID}, Int, Int}[]  # (tf, i_tlag, ι_lag_idx placeholder)
        for (tf, _) in state_lookup
            haskey(ι_lag, tf) || continue
            t, f = tf
            tlag = tlag_of(t)
            haskey(state_lookup, (tlag, f)) || continue
            push!(eligible, (tf, state_lookup[(tlag, f)], length(eligible) + 1))
        end

        n = length(eligible)
        ncols_state = size(state_matrix, 2)
        # Layout: [state_matrix[t-1], quarter_of_year(t), ι_lag]. Same column
        # order X_agg is built with in scripts/compliance_estimation.jl.
        X = Matrix{Float32}(undef, n, ncols_state + 2)
        for r in 1:n
            tf, i_tlag, _ = eligible[r]
            (t_year, t_q), _ = tf
            X[r, 1:ncols_state] .= @view state_matrix[i_tlag, :]
            X[r, ncols_state + 1] = Float32(t_q)
            X[r, end] = ι_lag[tf]
        end
        b = XGBoost.Booster(XGBoost.DMatrix(zeros(Float32, 1, ncols_state + 2)))
        XGBoost.load!(b, _sales_bytes)
        dm = XGBoost.DMatrix(X)
        logphi = XGBoost.predict(b, dm; margin = true)
        finalize(dm)
        finalize(b)

        out = Dict{Tuple{QuarterKey, UUID}, Float32}()
        for r in 1:n
            tf, _, _ = eligible[r]
            out[tf] = Float32(logphi[r])
        end
        return out
    end

    @memoize function j_to_tf_lookup()
        # Iterate the model's tf_to_j_lookup but key row indices off `_rd.tf_keys`
        # (persistent-eviction row layout). Extra (t, f) entries in `base` that
        # aren't in the residual-demand ordering are dropped along with (t, f)
        # pairs that lack a log-φ prediction. `kept_set` restricts js to the
        # canonical product universe.
        base = Supply.tf_to_j_lookup(model)
        idx_dict = Dict(k => i for (i, k) in enumerate(_rd.tf_keys))
        kept_set = Set(_rd.kept_vks)
        φ_lookup = logphi_lookup()
        out = Dict{VehicleKey, Vector{Tuple{Tuple{QuarterKey, UUID}, Int64}}}()
        for (tf, js) in base
            haskey(idx_dict, tf) || continue
            haskey(φ_lookup, tf) || continue
            for j in js
                j ∈ kept_set || continue
                if haskey(out, j)
                    push!(out[j], (tf, idx_dict[tf]))
                else
                    out[j] = [(tf, idx_dict[tf])]
                end
            end
        end
        return out
    end

    X_matrix(j) = let φ_lookup = logphi_lookup()
        mapreduce(vcat, j_to_tf_lookup()[j]) do (tf, i)
            row = Matrix{Float32}(undef, 1, size(state_matrix, 2) + 1)
            row[1, 1:(end - 1)] .= @view state_matrix[i, :]
            row[1, end] = φ_lookup[tf]
            return row
        end
    end

    log_prices(j) =
        map(j_to_tf_lookup()[j]) do (tf, _)
            t, f = tf
            idx = model.product_lookup.tjf[(t, j, f)]
            return log(model.product_info[idx].p)
        end

    # Nonparametric conditional density via quantile-regression grid. Fits one
    # XGBoost booster per level in Supply.QUANTILE_LEVELS with pinball loss.
    # BBL draws log-prices by inverse-CDF sampling with linear interpolation
    # between adjacent quantile knots (winsorized at the outer tails). Each
    # booster's base_score is seeded to the marginal empirical quantile by
    # fit_quantile_grid so early rounds don't waste climbing to the right level.
    QUANTILE_PARAMS = merge(
        Supply.XGB_DEFAULT_PARAMS,
        (
            subsample = 0.8,
            max_depth = 8,
            min_child_weight = 5,
            eta = 0.05,
            lambda = 1.0,
        ),
    )

    function fit_prices(j)
        @info "Fitting" j
        # Products whose entire 6-quarter observation window has no lagged
        # state coverage (typical: the earliest model year where only its
        # final quarter falls inside the BLP window) never get added to
        # j_to_tf_lookup by the log-φ filter. Skip cleanly.
        if !haskey(j_to_tf_lookup(), j)
            @warn "No training rows for $j after log-φ lag filter; skipping"
            return nothing
        end
        X, y = X_matrix(j), log_prices(j)
        quantile_bytes = Supply.fit_quantile_grid(
            X, y;
            max_round = 2000,
            patience = 20,
            params = QUANTILE_PARAMS,
        )
        n_active = count(!isempty, quantile_bytes)
        @info "  fitted quantile grid" j n_quantiles = length(quantile_bytes) n_active
        return quantile_bytes
    end
end

function (@main)(_)
    # Kept-VK universe comes straight from residual_demand.jld2, which was
    # produced under the kept-trims filter — every VK here already cleared
    # the share cutoff, so no further filtering is needed.
    js = _rd.kept_vks
    @info "Kept VK universe" n_kept = length(js) n_all = length(model.product_lookup.j)

    fitted = pmap(fit_prices, js)
    # Layout per VK: one Vector{UInt8} per quantile level in Supply.QUANTILE_LEVELS,
    # in the same order. BBL sampler reads via Supply.sample_from_quantile_grid.
    out = Dict{VehicleKey, Vector{Vector{UInt8}}}()
    for (j, f) in zip(js, fitted)
        f === nothing && continue
        out[j] = f
    end
    @info "Fit price densities" n_kept = length(out) n_skipped = length(js) - length(out) n_quantiles = length(
        Supply.QUANTILE_LEVELS
    )
    JLD2.save_object("data/price_densities_stratified.jld2", out)
    return nothing
end
