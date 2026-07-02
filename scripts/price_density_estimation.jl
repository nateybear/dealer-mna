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
    state_matrix = JLD2.load_object("data/residual_demand.jld2")

    # Sales-target booster + τ (from data_sales_target_estimation) — used to
    # compute log φ_{t,f} as an additional conditioning feature for prices.
    _sales_bytes, τ_sales = JLD2.load_object("data/sales_targets.jld2")

    # Precompute log φ_{t,f} for every (t, f) where the sales-target booster's
    # inputs are available (state_{t−1, f} in state_matrix and p_compliant at
    # t−1 in compliance.arrow). Rows without a lagged compliance entry drop
    # out — dealer's first observed quarter.
    @memoize function logphi_lookup()
        state_lookup = Dict(k => i for (i, k) in enumerate(keys(Supply.tf_to_j_lookup(model))))
        tlag_of(t) = t[2] == 1 ? (t[1] - 1, 4) : (t[1], t[2] - 1)

        # Sales-target booster was trained with ι_lag as a hard 1{p_compliant > 0.5}
        # indicator — use the same threshold here so the input matches training.
        compliance = read_arrow("data/compliance.arrow")
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
        X = Matrix{Float32}(undef, n, ncols_state + 1)
        for r in 1:n
            tf, i_tlag, _ = eligible[r]
            X[r, 1:ncols_state] .= @view state_matrix[i_tlag, :]
            X[r, end] = ι_lag[tf]
        end
        b = XGBoost.Booster(XGBoost.DMatrix(zeros(Float32, 1, ncols_state + 1)))
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
        base = Supply.tf_to_j_lookup(model)
        idx_dict = Dict(k => i for (i, k) in enumerate(keys(base)))
        φ_lookup = logphi_lookup()
        out = Dict{VehicleKey, Vector{Tuple{Tuple{QuarterKey, UUID}, Int64}}}()
        for (tf, js) in base
            # Drop (t, f) pairs where log φ_{t, f} is unavailable — dealer's
            # first observed quarter has no lagged compliance to feed the
            # sales-target booster.
            haskey(φ_lookup, tf) || continue
            for j in js
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

    # OOS-best-round + full-data refit. Identity link via reg:squarederror; mean
    # NLL is MSE which Supply.fit_xgb_oos uses as the default val_loss. Set
    # base_score = mean(y) so the booster starts at the right scale (log-prices
    # are ~9–11, abs residuals ~0.1–0.3); the C default of 0.5 wastes early
    # rounds climbing to the mean.
    regress(X, y) =
        Supply.fit_xgb_oos(
            X,
            y;
            max_round = 1000,
            patience = 5,
            params = (subsample = 0.8, base_score = Float64(mean(y))),
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
        mean_booster, mean_round = regress(X, y)
        y_hat = XGBoost.predict(mean_booster, X)
        # Squared residuals → variance estimate. Draw time takes sqrt of the
        # (clipped-nonneg) prediction to recover σ. More principled than the
        # abs-residual proxy: E[(y − ŷ)²|X] = Var(y|X) under an unbiased ŷ.
        u = (y .- y_hat) .^ 2
        variance_booster, variance_round = regress(X, u)
        @info "  best rounds" j mean_round variance_round
        # `XGBoost.save` returns an unsafe_wrap over libxgboost-owned memory;
        # copy so the bytes survive booster GC once fit_prices returns.
        return copy(XGBoost.save(mean_booster, Vector{UInt8}; format = "ubj")),
        copy(XGBoost.save(variance_booster, Vector{UInt8}; format = "ubj"))
    end
end

function (@main)(_)
    js = collect(keys(model.product_lookup.j))
    fitted = pmap(fit_prices, js)
    out = Dict{VehicleKey, Tuple{Vector{UInt8}, Vector{UInt8}}}()
    for (j, f) in zip(js, fitted)
        f === nothing && continue
        out[j] = f
    end
    @info "Fit price densities" n_kept = length(out) n_skipped = length(js) - length(out)
    JLD2.save_object("data/price_densities.jld2", out)
    return nothing
end
