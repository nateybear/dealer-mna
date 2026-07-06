using Distributed
@everywhere using Memoize
@everywhere using StatsBase: ecdf, mean, quantile, var, cov

@everywhere begin
    using DealerMnA
    using XGBoost
    using LoggingExtras
    using LinearAlgebra
    using LogExpFunctions
    using StatsBase: mean, cov, quantile
    import .Supply: selectunique

    BLAS.set_num_threads(1)

    DEMOS = Empirics.DEFAULT_DEMOGRAPHICS

    # K-fold cross-fitting for honest p_base and ρ̃; both share the fold map.
    K = 3

    # Alternating fits of (p_base, ρ̃) within each M-step fold. Iterate until
    # ‖Δρ̃_tr‖_∞ < tolerance, capped by MAX_INNER_SWEEPS.
    MAX_INNER_SWEEPS = 20
    Ρ̃_TOL = 1.0f-3

    # Number of boosting rounds in the step-1 cross-fit. Step 2's final refit
    # picks its own num_round per booster via Supply.fit_xgb_custom_oos.
    NUM_ROUND_STEP1 = 6

    # Numeric floor for log(π) / log(1-π) in the E-step tempered log-odds.
    Π_FLOOR = 1.0f-3

    # ρ̃ booster overrides base_score to −6 (σ(−6) ≈ 0.0025). XGB_DEFAULT_PARAMS
    # uses −2 (σ(−2) ≈ 0.12), which is a reasonable prior for a sales rate but
    # not for ρ̃: the trivial local optimum of the mixture EM sits at ρ̃ ≡ 0
    # (complier branch collapses onto noncomplier), so the "no signal → base"
    # default for ρ̃ should be near 0, not 0.12. Otherwise the complier branch
    # systematically over-predicts sales and drags γ toward 0.
    RHO_XGB_PARAMS = merge(Supply.XGB_DEFAULT_PARAMS, (base_score = -6.0,))

    demand_model = Demand.DemandModel(; JLD2.load_object("data/model_frames.jld2")...)

    state_matrix = JLD2.load_object("data/residual_demand.jld2")

    # Whitens dealer demographics. p_base no longer reads these — kept so the
    # demographic columns retain their conventional standardized form for any
    # downstream consumer.
    function prepare_dealers!(dealer_demographics)
        M = Matrix(dealer_demographics[!, DEMOS])
        M .-= mean(M, dims = 1)
        M *= sqrt(inv(cov(M)))
        dealer_demographics[!, DEMOS] .= M
        return nothing
    end

    @memoize function state_lookup()
        intermed = Supply.tf_to_j_lookup(demand_model)
        return Dict(k => i for (i, k) in enumerate(keys(intermed)))
    end

    function X_matrix(cars)
        nrows = nrow(cars)
        ncols = size(state_matrix, 2)
        X = Matrix{Float32}(undef, nrows, ncols)
        lookup = state_lookup()
        for (i, c) in enumerate(eachrow(cars))
            quarter = (Dates.year(c.t), Dates.quarterofyear(c.t))
            X[i, :] .= @view state_matrix[lookup[(quarter, c.dealer_uid)], :]
        end
        return X
    end

    function log_e_step_diagnostics(E_results)
        qs = [0.1, 0.5, 0.9]
        p_compliant = [r.p_compliant for r in E_results]
        ll_gap = [r.ll_complier - r.ll_noncomplier for r in E_results]
        ρ̃_means = [r.ρ̃_mean for r in E_results]
        p_base_means = [r.p_base_mean for r in E_results]
        n_cars_vec = [r.n_cars for r in E_results]
        obs_means = [r.obs_mean for r in E_results]
        obs_vars = [r.obs_var for r in E_results]
        nc_var_pred = @. p_base_means * (1 - p_base_means)
        sale_lift = obs_means .- p_base_means

        @info "  ll_complier - ll_noncomplier per (d,t)" mean = mean(ll_gap) q = quantile(
            ll_gap, qs
        ) n_positive = sum(>(0), ll_gap) total = length(ll_gap)
        @info "  γ = p_compliant" mean = mean(p_compliant) q = quantile(p_compliant, qs) frac_above_0_9 = mean(
            >(0.9), p_compliant
        ) frac_below_0_1 = mean(<(0.1), p_compliant)
        @info "  ρ̃ (mean per d,t)" mean = mean(ρ̃_means) q = quantile(ρ̃_means, qs)
        @info "  p_base (mean per d,t)" mean = mean(p_base_means) q = quantile(p_base_means, qs)
        @info "  cars per (d,t)" mean = mean(n_cars_vec) q = quantile(n_cars_vec, qs)
        @info "  within-(d,t) variance" obs = mean(obs_vars) nc_pred = mean(nc_var_pred) frac_obs_above_nc = mean(
            obs_vars .> nc_var_pred
        )
        @info "  observed sale rate vs p_base" obs = mean(obs_means) p_base = mean(p_base_means) lift_mean = mean(
            sale_lift
        ) lift_q = quantile(sale_lift, qs)
        return nothing
    end
end

function (@main)(_)
    Dealers.load_from_disk!("data/dealers_init_filtered.jld2")
    fords = read_arrow("data/arrow/fords.arrow")
    dealer_demographics = semijoin(
        read_arrow("data/dealer_demographics.arrow"), fords, on = :uid => :dealer_uid
    )
    prepare_dealers!(dealer_demographics)

    panel = Supply.make_panel(fords, dealer_demographics)
    select!(panel, :dealer_uid, :t, :model, :model_year, :extra, :is_sold)
    subset!(panel, :t => ByRow(≥(Date(2017))))

    compliance = selectunique(panel, [:dealer_uid, :t])

    # Initialize γ_dt as the empirical CDF value of next-quarter sales count:
    # cells whose dealer sells many cars at t+1 get high γ_dt, cells with low
    # next-quarter sales get low γ_dt. Compliance persists across quarters, so
    # next-quarter sales are a leading indicator of current-quarter compliance
    # — and, crucially, γ now carries state-informative variation from round 1,
    # which is what the M-step needs to fit distinct p_base(x) and ρ̃(x).
    # Cells at the last observed quarter (no t+1) fall back to γ = 0.5.
    sales_by_dt = combine(groupby(panel, [:dealer_uid, :t]), :is_sold => sum => :n_sold)
    sales_lookup = Dict((r.dealer_uid, r.t) => r.n_sold for r in eachrow(sales_by_dt))
    next_sold = Float32[
        get(sales_lookup, (r.dealer_uid, r.t + Quarter(1)), NaN32) for
        r in eachrow(compliance)
    ]
    mask = .!isnan.(next_sold)
    F_next = ecdf(next_sold[mask])
    compliance.p_compliant = fill(0.5f0, nrow(compliance))
    compliance.p_compliant[mask] .= Float32.(F_next.(next_sold[mask]))
    @info "γ init from next-quarter sales" mean_γ = mean(compliance.p_compliant) frac_last_quarter = 1 -
        sum(mask) / length(mask)

    # global fold assignment so each (dealer, t) is in the same fold across
    # every vehicle's p_base and ρ̃ boosters and matches its E-step prediction.
    fold_map = Dict(
        (row.dealer_uid, row.t) => mod(hash((row.dealer_uid, row.t)), K) + 1 for
        row in eachrow(compliance)
    )

    gdf = groupby(panel, [:model, :model_year, :extra])

    M_step = Dict()
    E_step = empty!(copy(compliance))

    for i in 1:100
        @info "Starting round $i"
        @everywhere GC.gc(true)
        π = clamp(mean(compliance.p_compliant), Π_FLOOR, 1 - Π_FLOOR)
        M_step =
            let mapped = pmap(collect(gdf)) do cars
                    cars = leftjoin(cars, compliance, on = [:dealer_uid, :t])
                    X = X_matrix(cars)
                    y_all = Float32.(cars.is_sold)
                    γ_all = Float32.(cars.p_compliant)
                    folds = [fold_map[(c.dealer_uid, c.t)] for c in eachrow(cars)]

                    pbase_bytes = Vector{Vector{UInt8}}(undef, K)
                    rho_bytes = Vector{Vector{UInt8}}(undef, K)
                    sweep_counts = Vector{Int}(undef, K)

                    for k in 1:K
                        train_idx = findall(!=(k), folds)
                        X_tr = X[train_idx, :]
                        y_tr = y_all[train_idx]
                        γ_tr = γ_all[train_idx]
                        n_tr = length(train_idx)

                        dm_tr = XGBoost.DMatrix(X_tr)
                        XGBoost.setinfo!(dm_tr, "label", y_tr)

                        ρ̃_tr = zeros(Float32, n_tr)
                        saved_p_bytes = UInt8[]
                        saved_r_bytes = UInt8[]
                        last_sweep = 0

                        for sweep in 1:MAX_INNER_SWEEPS
                            last_sweep = sweep
                            ρ̃_prev = copy(ρ̃_tr)

                            booster_p = Supply.fit_xgb_custom(
                                dm_tr; num_round = NUM_ROUND_STEP1
                            ) do ŷ
                                grad = Vector{Float32}(undef, n_tr)
                                hess = Vector{Float32}(undef, n_tr)
                                @inbounds for j in 1:n_tr
                                    grad[j], hess[j] = Supply.pbase_grad_hess(
                                        ŷ[j], y_tr[j], γ_tr[j], ρ̃_tr[j]
                                    )
                                end
                                return grad, hess
                            end
                            p_tr =
                                logistic.(XGBoost.predict(booster_p, dm_tr; margin = true))
                            saved_p_bytes = copy(
                                XGBoost.save(booster_p, Vector{UInt8}; format = "ubj")
                            )
                            finalize(booster_p)

                            # ρ̃ fit: every row contributes with weight γ; p_base held fixed
                            booster_r = Supply.fit_xgb_custom(
                                dm_tr; num_round = NUM_ROUND_STEP1, params = RHO_XGB_PARAMS
                            ) do ŷ
                                grad = Vector{Float32}(undef, n_tr)
                                hess = Vector{Float32}(undef, n_tr)
                                @inbounds for j in 1:n_tr
                                    grad[j], hess[j] = Supply.rho_grad_hess(
                                        ŷ[j], y_tr[j], γ_tr[j], p_tr[j]
                                    )
                                end
                                return grad, hess
                            end
                            ρ̃_tr .=
                                logistic.(XGBoost.predict(booster_r, dm_tr; margin = true))
                            saved_r_bytes = copy(
                                XGBoost.save(booster_r, Vector{UInt8}; format = "ubj")
                            )
                            finalize(booster_r)

                            maximum(abs, ρ̃_tr .- ρ̃_prev) < Ρ̃_TOL && break
                        end

                        pbase_bytes[k] = saved_p_bytes
                        rho_bytes[k] = saved_r_bytes
                        sweep_counts[k] = last_sweep
                        finalize(dm_tr)
                    end

                    (; model, model_year, extra) = cars[1, :]
                    return (model, model_year, extra) =>
                        (pbase_bytes, rho_bytes, sweep_counts)
                end
                mapreduce(
                    Dict,
                    merge,
                    mapped,
                    init = Dict{
                        Tuple{String, Int64, String},
                        Tuple{
                            Vector{Vector{UInt8}}, Vector{Vector{UInt8}}, Vector{Int}
                        },
                    }(),
                )
            end
        all_sweeps = reduce(vcat, [v[3] for v in values(M_step)])
        @info "M-step inner sweeps" mean = mean(all_sweeps) q = quantile(
            all_sweeps, [0.1, 0.5, 0.9]
        ) hit_cap = sum(==(MAX_INNER_SWEEPS), all_sweeps) total = length(all_sweeps)

        @info "Starting E step"

        E_step = copy(compliance)
        E_results = pmap(eachrow(E_step); batch_size = 100) do r
            cars = subset(panel, :dealer_uid => ByRow(==(r.dealer_uid)), :t => ByRow(==(r.t)))
            k = fold_map[(r.dealer_uid, r.t)]

            # All cars at this (d,t) share the same state row, so a single-row
            # DMatrix suffices for every per-vehicle booster prediction.
            dm = XGBoost.DMatrix(X_matrix(cars[[1], :]))
            booster = XGBoost.Booster(dm)

            cars.p_base = map(eachrow(cars)) do (; model, model_year, extra)
                pbase_bytes = M_step[(model, model_year, extra)][1][k]
                XGBoost.load!(booster, pbase_bytes)
                f_raw = only(XGBoost.predict(booster, dm; margin = true))
                clamp(logistic(f_raw), 1e-12, 1 - 1e-12)
            end

            cars.ρ_tilde = map(eachrow(cars)) do (; model, model_year, extra)
                rho_bytes = M_step[(model, model_year, extra)][2][k]
                XGBoost.load!(booster, rho_bytes)
                g_raw = only(XGBoost.predict(booster, dm; margin = true))
                clamp(logistic(g_raw), 1e-12, 1 - 1e-12)
            end

            finalize(booster)
            finalize(dm)

            ll_complier = let (; is_sold, p_base, ρ_tilde) = cars
                sum(
                    @. is_sold * log(p_base + (1 - p_base) * ρ_tilde) +
                        (1 - is_sold) * log((1 - p_base) * (1 - ρ_tilde))
                )
            end

            ll_noncomplier = let (; is_sold, p_base) = cars
                sum(@. is_sold * log(p_base) + (1 - is_sold) * log(1 - p_base))
            end
            p_compliant = softmax(
            [log(π) + ll_complier, log(1 - π) + ll_noncomplier]
)[1]
            return (;
                p_compliant,
                ll_complier,
                ll_noncomplier,
                ρ̃_mean = mean(cars.ρ_tilde),
                p_base_mean = mean(cars.p_base),
                n_cars = nrow(cars),
                obs_mean = mean(cars.is_sold),
                obs_var = var(cars.is_sold),
            )
        end
        E_step.p_compliant = [r.p_compliant for r in E_results]
        log_e_step_diagnostics(E_results)

        step_change = norm(E_step.p_compliant - compliance.p_compliant)
        max_change = maximum(abs, E_step.p_compliant - compliance.p_compliant)
        @info "Round $i" step_change max_change π mean_p_compliant = mean(
            E_step.p_compliant
        )

        # Convergence only counts once the temperature has fully annealed to 1;
        # earlier rounds are fitting a smoothed likelihood, not the target one.
        if max_change < 1.0f-4
            @info "EM converged at step $i (max Δγ = $max_change)"
            Arrow.write("data/compliance.arrow", E_step)
            JLD2.save_object("data/compliance_fits.jld2", M_step)

            @info "Step 2: full-data refit with OOS early stopping"
            compliance.p_compliant .= E_step.p_compliant
            final_M_step =
                let mapped = pmap(collect(gdf)) do cars
                        cars = leftjoin(cars, compliance, on = [:dealer_uid, :t])
                        X = X_matrix(cars)
                        y_all = Float32.(cars.is_sold)
                        γ_all = Float32.(cars.p_compliant)
                        n = length(y_all)

                        ρ̃ = zeros(Float32, n)
                        pbase_bytes = UInt8[]
                        rho_bytes = UInt8[]
                        best_round_p = 0
                        best_round_r = 0

                        for sweep in 1:MAX_INNER_SWEEPS
                            ρ̃_prev = copy(ρ̃)

                            booster_p, br_p = Supply.fit_xgb_custom_oos(
                                idxs ->
                                    let m = length(idxs),
                                        y_s = y_all[idxs],
                                        γ_s = γ_all[idxs],
                                        ρ̃_s = ρ̃[idxs]

                                        ŷ -> begin
                                            grad = Vector{Float32}(undef, m)
                                            hess = Vector{Float32}(undef, m)
                                            @inbounds for j in 1:m
                                                grad[j], hess[j] = Supply.pbase_grad_hess(
                                                    ŷ[j], y_s[j], γ_s[j], ρ̃_s[j]
                                                )
                                            end
                                            return grad, hess
                                        end
                                    end,
                                idxs ->
                                    let m = length(idxs),
                                        y_s = y_all[idxs],
                                        γ_s = γ_all[idxs],
                                        ρ̃_s = ρ̃[idxs]

                                        ŷ -> begin
                                            nll = 0.0
                                            @inbounds for j in 1:m
                                                nll += Supply.pbase_nll(
                                                    ŷ[j], y_s[j], γ_s[j], ρ̃_s[j]
                                                )
                                            end
                                            return nll / m
                                        end
                                    end,
                                X,
                                y_all;
                                max_round = 50,
                                patience = 5,
                            )
                            pbase_bytes = copy(
                                XGBoost.save(booster_p, Vector{UInt8}; format = "ubj")
                            )
                            best_round_p = br_p

                            dm_full = XGBoost.DMatrix(X)
                            p = logistic.(XGBoost.predict(booster_p, dm_full; margin = true))
                            finalize(booster_p)

                            booster_r, br_r = Supply.fit_xgb_custom_oos(
                                idxs ->
                                    let m = length(idxs),
                                        y_s = y_all[idxs],
                                        γ_s = γ_all[idxs],
                                        p_s = p[idxs]

                                        ŷ -> begin
                                            grad = Vector{Float32}(undef, m)
                                            hess = Vector{Float32}(undef, m)
                                            @inbounds for j in 1:m
                                                grad[j], hess[j] = Supply.rho_grad_hess(
                                                    ŷ[j], y_s[j], γ_s[j], p_s[j]
                                                )
                                            end
                                            return grad, hess
                                        end
                                    end,
                                idxs ->
                                    let m = length(idxs),
                                        y_s = y_all[idxs],
                                        γ_s = γ_all[idxs],
                                        p_s = p[idxs]

                                        ŷ -> begin
                                            nll = 0.0
                                            @inbounds for j in 1:m
                                                nll += Supply.rho_nll(
                                                    ŷ[j], y_s[j], γ_s[j], p_s[j]
                                                )
                                            end
                                            return nll / m
                                        end
                                    end,
                                X,
                                y_all;
                                max_round = 50,
                                patience = 5,
                                params = RHO_XGB_PARAMS,
                            )
                            ρ̃ .= logistic.(XGBoost.predict(booster_r, dm_full; margin = true))
                            rho_bytes = copy(XGBoost.save(booster_r, Vector{UInt8}; format = "ubj"))
                            best_round_r = br_r
                            finalize(booster_r)
                            finalize(dm_full)

                            maximum(abs, ρ̃ .- ρ̃_prev) < Ρ̃_TOL && break
                        end

                        (; model, model_year, extra) = cars[1, :]
                        @info "  refit" model model_year extra best_round_p best_round_r
                        return (model, model_year, extra) => (pbase_bytes, rho_bytes)
                    end
                    mapreduce(
                        Dict,
                        merge,
                        mapped,
                        init = Dict{
                            Tuple{String, Int64, String},
                            Tuple{Vector{UInt8}, Vector{UInt8}},
                        }(),
                    )
                end
            JLD2.save_object("data/compliance_fits_final.jld2", final_M_step)
            return nothing
        end
        compliance.p_compliant .= E_step.p_compliant
    end
    @error "Boohoo, no convergence after 100 steps"
    return nothing
end
