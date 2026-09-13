using Distributed
@everywhere using Memoize
@everywhere using StatsBase: ecdf, mean, quantile, std, var, cov

@everywhere begin
    using DealerMnA
    using XGBoost
    using LoggingExtras
    using LinearAlgebra
    using LogExpFunctions
    using Random
    using StatsBase: mean, cov, quantile
    using Distributions: Bernoulli, Beta
    import .Supply: selectunique

    BLAS.set_num_threads(1)

    DEMOS = Empirics.DEFAULT_DEMOGRAPHICS

    K = 3

    MAX_INNER_SWEEPS = 100_000
    Ρ̃_TOL = 1.0f-4
    NUM_ROUND_STEP1 = 20

    # Bias the sweep-0 p_base fit toward "compliers have a real lift" so p_base
    # doesn't collapse onto ȳ and leave ρ̃ with the (ȳ − p_base)/(1 − p_base) = 0
    # fixed point.
    Ρ̃_INIT = 0.15f0

    Π_FLOOR = 1.0f-3

    N_TOTAL = 100
    BURN_IN = 30

    RHO_XGB_PARAMS = merge(
        Supply.XGB_DEFAULT_PARAMS,
        (base_score = -3.0, lambda = 5.0f-1, max_depth = 4, eta = 0.1),
    )

    PBASE_XGB_PARAMS = merge(
        Supply.XGB_DEFAULT_PARAMS,
        (max_depth = 4, eta = 0.1),
    )

    Z_CLAMP = 0.05f0

    demand_model = Demand.DemandModel(; JLD2.load_object("data/model_frames.jld2")...)

    state_matrix = JLD2.load_object("data/residual_demand.jld2")

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
            tlag = c.t - Quarter(1)
            quarter = (Dates.year(tlag), Dates.quarterofyear(tlag))
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
        phi_contribs = [r.phi_contrib for r in E_results]
        active = filter(!=(0.0), phi_contribs)
        @info "  φ log-odds contribution (q − φ)/τ" mean = mean(phi_contribs) q = quantile(
            phi_contribs, qs
        ) active_mean = isempty(active) ? 0.0 : mean(active) n_active = length(active) total = length(
            phi_contribs
        )
        return nothing
    end
end

A_PRIOR=10

function (@main)(_)
    Dealers.load_from_disk!("data/dealers_init_filtered.jld2")
    fords = read_arrow("data/arrow/fords.arrow")

    kept_pairs = Set(
        (row.model, row.extra) for
        row in eachrow(read_arrow("data/kept_trims.arrow"))
    )
    fords = subset(
        fords,
        [:model, :extra] => ByRow((m, e) -> (m, e) ∈ kept_pairs),
    )
    # Must match bbl_forward_simulation.jl's max_model_year (T runs 2017-2019).
    fords = subset(fords, :model_year => ByRow(≤(2019)))
    @info "Filtered fords" n_rows = nrow(fords) n_kept_trims = length(kept_pairs)

    dealer_demographics = semijoin(
        read_arrow("data/dealer_demographics.arrow"), fords, on = :uid => :dealer_uid
    )
    prepare_dealers!(dealer_demographics)

    panel = Supply.make_panel(fords, dealer_demographics)
    select!(panel, :dealer_uid, :t, :model, :model_year, :extra, :is_sold)
    subset!(panel, :t => ByRow(≥(Date(2017))))

    subset!(
        panel,
        [:t, :dealer_uid] => ByRow(
            (t, f) -> haskey(
                state_lookup(),
                ((Dates.year(t - Quarter(1)), Dates.quarterofyear(t - Quarter(1))), f),
            ),
        ),
    )

    compliance = selectunique(panel, [:dealer_uid, :t])

    dq = select(Supply.make_dealer_quarter(fords), :dealer_uid, :t, :total_units => :q)
    agg_master = innerjoin(compliance, dq, on = [:dealer_uid, :t])
    subset!(
        agg_master,
        [:t, :dealer_uid] => ByRow(
            (t, f) -> haskey(
                state_lookup(),
                ((Dates.year(t - Quarter(1)), Dates.quarterofyear(t - Quarter(1))), f),
            ),
        ),
    )
    select!(agg_master, :dealer_uid, :t, :q)

    n_agg = nrow(agg_master)
    ncols_state = size(state_matrix, 2)
    X_agg_base = Matrix{Float32}(undef, n_agg, ncols_state + 1)
    for (i, r) in enumerate(eachrow(agg_master))
        tlag = r.t - Quarter(1)
        quarter = (Dates.year(tlag), Dates.quarterofyear(tlag))
        X_agg_base[i, 1:ncols_state] .= @view state_matrix[
            state_lookup()[(quarter, r.dealer_uid)], :,
        ]
        X_agg_base[i, end] = Float32(Dates.quarterofyear(r.t))
    end
    q_agg = Float32.(agg_master.q)

    τ = Float32(0.5 * std(q_agg))
    @info "τ fixed for sales-target booster" τ mean_q = mean(q_agg) std_q = std(q_agg) n_agg_rows =
        n_agg

    PHI_XGB_PARAMS = merge(
        Supply.XGB_DEFAULT_PARAMS,
        (
            base_score = Float64(log(mean(q_agg))),
            max_depth = 4,
            min_child_weight = 1,
            lambda = 1.0,
            eta = 0.1,
        ),
    )

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
    @info "γ init from next-quarter sales" mean_γ = mean(compliance.p_compliant) frac_last_quarter =
        1 -
        sum(mask) / length(mask)

    gdf = groupby(panel, [:model, :model_year, :extra])

    M_step = Dict()
    E_step = empty!(copy(compliance))

    compliance.p_compliant .= Float32.(
        rand.(Bernoulli.(clamp.(compliance.p_compliant, Z_CLAMP, 1 - Z_CLAMP)))
    )

    γ_accum = zeros(Float64, nrow(compliance))
    π_samples = Float64[]
    n_kept = 0

    φ̂_accum = zeros(Float64, n_agg)
    phi_booster_bytes = UInt8[]

    pi_trace = Vector{Float64}(undef, N_TOTAL)
    gamma_trace = Matrix{Float32}(undef, N_TOTAL, nrow(compliance))
    phi_hat_trace = Matrix{Float32}(undef, N_TOTAL, n_agg)

    for i in 1:N_TOTAL
        @info "Starting round $i"
        @everywhere GC.gc(true)
        π = clamp(
            rand(
                Beta(
                    Float64(A_PRIOR + sum(compliance.p_compliant)),
                    Float64(A_PRIOR + nrow(compliance) - sum(compliance.p_compliant)),
                ),
            ),
            Float64(Π_FLOOR),
            1 - Float64(Π_FLOOR),
        )
        M_step =
            let mapped = pmap(collect(gdf)) do cars
                    cars = leftjoin(cars, compliance, on = [:dealer_uid, :t])
                    X = X_matrix(cars)
                    y_all = Float32.(cars.is_sold)
                    γ_all = Float32.(cars.p_compliant)

                    (; model, model_year, extra) = cars[1, :]
                    rng_trim = Xoshiro(hash((model, model_year, extra)))
                    folds = Supply._stratified_kfold(y_all, K, rng_trim)

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

                        ρ̃_tr = fill(Ρ̃_INIT, n_tr)
                        saved_p_bytes = UInt8[]
                        saved_r_bytes = UInt8[]
                        last_sweep = 0

                        for sweep in 1:MAX_INNER_SWEEPS
                            last_sweep = sweep
                            ρ̃_prev = copy(ρ̃_tr)

                            booster_p = Supply.fit_xgb_custom(
                                dm_tr; num_round = NUM_ROUND_STEP1,
                                params = PBASE_XGB_PARAMS,
                                xgb_model = saved_p_bytes,
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
                            p_tr = logistic.(XGBoost.predict(booster_p, dm_tr; margin = true))
                            saved_p_bytes = copy(
                                XGBoost.save(booster_p, Vector{UInt8}; format = "ubj")
                            )
                            finalize(booster_p)

                            complier_local = findall(==(1.0f0), γ_tr)
                            if isempty(complier_local)
                                saved_r_bytes = UInt8[]
                                fill!(ρ̃_tr, Supply.RHO_MIN)
                            else
                                X_tr_c = X_tr[complier_local, :]
                                y_tr_c = y_tr[complier_local]
                                p_tr_c = p_tr[complier_local]
                                n_tr_c = length(complier_local)
                                dm_tr_c = XGBoost.DMatrix(X_tr_c)
                                XGBoost.setinfo!(dm_tr_c, "label", y_tr_c)

                                booster_r = Supply.fit_xgb_custom(
                                    dm_tr_c;
                                    num_round = NUM_ROUND_STEP1,
                                    params = RHO_XGB_PARAMS,
                                    xgb_model = saved_r_bytes,
                                ) do ŷ
                                    grad = Vector{Float32}(undef, n_tr_c)
                                    hess = Vector{Float32}(undef, n_tr_c)
                                    @inbounds for j in 1:n_tr_c
                                        grad[j], hess[j] = Supply.rho_grad_hess(
                                            ŷ[j], y_tr_c[j], 1.0f0, p_tr_c[j]
                                        )
                                    end
                                    return grad, hess
                                end
                                fill!(ρ̃_tr, Supply.RHO_MIN)
                                ρ̃_tr[complier_local] .= Supply.rho_from_margin.(
                                    XGBoost.predict(booster_r, dm_tr_c; margin = true)
                                )
                                saved_r_bytes = copy(
                                    XGBoost.save(booster_r, Vector{UInt8}; format = "ubj")
                                )
                                finalize(booster_r)
                                finalize(dm_tr_c)
                            end

                            maximum(abs, ρ̃_tr .- ρ̃_prev) < Ρ̃_TOL && break
                        end

                        pbase_bytes[k] = saved_p_bytes
                        rho_bytes[k] = saved_r_bytes
                        sweep_counts[k] = last_sweep
                        finalize(dm_tr)
                    end

                    cell_fold_map = Dict{Tuple{UUID, Date}, Int}(
                        (c.dealer_uid, c.t) => folds[i] for
                        (i, c) in enumerate(eachrow(cars))
                    )
                    return (model, model_year, extra) =>
                        (pbase_bytes, rho_bytes, sweep_counts, cell_fold_map)
                end
                mapreduce(
                    Dict,
                    merge,
                    mapped,
                    init = Dict{
                        Tuple{String, Int64, String},
                        Tuple{
                            Vector{Vector{UInt8}}, Vector{Vector{UInt8}}, Vector{Int},
                            Dict{Tuple{UUID, Date}, Int},
                        },
                    }(),
                )
            end
        all_sweeps = reduce(vcat, [v[3] for v in values(M_step)])
        @info "M-step inner sweeps" mean = mean(all_sweeps) q = quantile(
            all_sweeps, [0.1, 0.5, 0.9]
        ) hit_cap = sum(==(MAX_INNER_SWEEPS), all_sweeps) total = length(all_sweeps)

        sampled_lookup = Dict(
            (r.dealer_uid, r.t) => r.p_compliant for r in eachrow(compliance)
        )
        ι_agg = Float32[
            Float32(get(sampled_lookup, (r.dealer_uid, r.t), 0.0f0)) for
            r in eachrow(agg_master)
        ]
        ι_lag_agg = Float32[
            Float32(get(sampled_lookup, (r.dealer_uid, r.t - Quarter(1)), 0.0f0) > 0.5f0) for
            r in eachrow(agg_master)
        ]
        X_agg = hcat(X_agg_base, ι_lag_agg)
        dm_agg = XGBoost.DMatrix(X_agg)
        y_agg = @. (2 * ι_agg - 1) * q_agg
        XGBoost.setinfo!(dm_agg, "label", y_agg)
        phi_booster = Supply.fit_xgb_custom(
            dm_agg; num_round = NUM_ROUND_STEP1, params = PHI_XGB_PARAMS
        ) do ŷ
            grad = Vector{Float32}(undef, n_agg)
            hess = Vector{Float32}(undef, n_agg)
            @inbounds for j in 1:n_agg
                grad[j], hess[j] = Supply.phi_grad_hess(ŷ[j], q_agg[j], ι_agg[j], τ)
            end
            return grad, hess
        end
        φ_pred = exp.(XGBoost.predict(phi_booster, dm_agg; margin = true))
        phi_booster_bytes = copy(XGBoost.save(phi_booster, Vector{UInt8}; format = "ubj"))
        finalize(phi_booster)
        finalize(dm_agg)
        φ_lookup = Dict(
            (agg_master.dealer_uid[j], agg_master.t[j]) => φ_pred[j] for j in 1:n_agg
        )
        q_lookup = Dict(
            (agg_master.dealer_uid[j], agg_master.t[j]) => q_agg[j] for j in 1:n_agg
        )
        @info "φ̂ quantiles" q = quantile(φ_pred, [0.1, 0.5, 0.9]) mean_lift = mean(
            @. (q_agg - φ_pred) / τ
        ) frac_ι_agg_one = mean(ι_agg) frac_ι_lag_one = mean(ι_lag_agg)

        @info "Starting E step"

        E_step = copy(compliance)
        E_step.q_agg = Float32[
            get(q_lookup, (r.dealer_uid, r.t), 0.0f0) for r in eachrow(E_step)
        ]
        E_step.phi_hat = Float32[
            get(φ_lookup, (r.dealer_uid, r.t), NaN32) for r in eachrow(E_step)
        ]
        E_results = pmap(eachrow(E_step); batch_size = 100) do r
            cars = subset(panel, :dealer_uid => ByRow(==(r.dealer_uid)), :t => ByRow(==(r.t)))

            dm = XGBoost.DMatrix(X_matrix(cars[[1], :]))
            booster = XGBoost.Booster(dm)

            cars.p_base = map(eachrow(cars)) do (; model, model_year, extra)
                trim_info = M_step[(model, model_year, extra)]
                k = trim_info[4][(r.dealer_uid, r.t)]
                pbase_bytes = trim_info[1][k]
                XGBoost.load!(booster, pbase_bytes)
                f_raw = only(XGBoost.predict(booster, dm; margin = true))
                return clamp(logistic(f_raw), 1e-12, 1 - 1e-12)
            end

            cars.ρ_tilde = map(eachrow(cars)) do (; model, model_year, extra)
                trim_info = M_step[(model, model_year, extra)]
                k = trim_info[4][(r.dealer_uid, r.t)]
                rho_bytes = trim_info[2][k]
                isempty(rho_bytes) && return Float64(Supply.RHO_MIN)
                XGBoost.load!(booster, rho_bytes)
                g_raw = only(XGBoost.predict(booster, dm; margin = true))
                return clamp(Supply.rho_from_margin(g_raw), 1e-12, 1 - 1e-12)
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
            phi_contrib = isnan(r.phi_hat) ? 0.0 : (r.q_agg - r.phi_hat) / τ
            p_compliant = softmax(
                [log(π) + ll_complier + phi_contrib, log(1 - π) + ll_noncomplier]
            )[1]
            return (;
                p_compliant,
                ll_complier,
                ll_noncomplier,
                phi_contrib,
                ρ̃_mean = mean(cars.ρ_tilde),
                p_base_mean = mean(cars.p_base),
                n_cars = nrow(cars),
                obs_mean = mean(cars.is_sold),
                obs_var = var(cars.is_sold),
            )
        end
        E_step.p_compliant = [r.p_compliant for r in E_results]
        log_e_step_diagnostics(E_results)

        pi_trace[i] = π
        gamma_trace[i, :] .= E_step.p_compliant
        phi_hat_trace[i, :] .= φ_pred

        if i > BURN_IN
            γ_accum .+= E_step.p_compliant
            push!(π_samples, π)
            φ̂_accum .+= φ_pred
            n_kept += 1
        end

        compliance.p_compliant .= Float32.(
            rand.(Bernoulli.(clamp.(E_step.p_compliant, Z_CLAMP, 1 - Z_CLAMP)))
        )

        @info "Round $i" π mean_γ = mean(E_step.p_compliant) frac_z_one = mean(
            compliance.p_compliant
        ) accumulated = n_kept

        if i == N_TOTAL
            posterior_φ̂ = Float32.(φ̂_accum ./ max(n_kept, 1))
            @info "Sampler complete: writing posterior-mean γ" n_kept mean_π = mean(
                π_samples
            ) std_π = std(π_samples) mean_φ̂ = mean(posterior_φ̂) q_φ̂ = quantile(
                posterior_φ̂, [0.1, 0.5, 0.9]
            )
            posterior_γ = Float32.(γ_accum ./ max(n_kept, 1))
            E_step.p_compliant = posterior_γ
            compliance.p_compliant .= posterior_γ
            Arrow.write("data/compliance_stratified.arrow", select(E_step, Not(:q_agg, :phi_hat)))
            JLD2.save_object("data/compliance_fits_stratified.jld2", M_step)
            JLD2.save_object("data/sales_targets_stratified.jld2", (phi_booster_bytes, τ))
            JLD2.save_object(
                "data/compliance_mcem_trace_stratified.jld2",
                (;
                    pi_trace,
                    gamma_trace,
                    phi_hat_trace,
                    dealer_uid = compliance.dealer_uid,
                    t = compliance.t,
                    agg_dealer_uid = agg_master.dealer_uid,
                    agg_t = agg_master.t,
                    N_TOTAL,
                    BURN_IN,
                ),
            )

            @info "Step 2: full-data refit mirroring the step-1 inner sweep"
            # Mirrors step 1's per-fold inner sweep verbatim, on full data with
            # a hard γ threshold (γ = 1{p_compliant > 0.5}). Warm-start across
            # sweeps via xgb_model = saved_*_bytes so the saved booster
            # accumulates trees exactly the way step 1's cross-fit boosters do —
            # the previous fit_xgb_custom_oos(max_round=50, patience=5) capped
            # every saved booster at one fresh ~50-round fit, which under-fit
            # per-group carry probabilities and collapsed sim assortments toward
            # the marginal mean.
            final_M_step =
                let mapped = pmap(collect(gdf)) do cars
                        cars = leftjoin(cars, compliance, on = [:dealer_uid, :t])
                        X = X_matrix(cars)
                        y_all = Float32.(cars.is_sold)
                        γ_all = Float32.(cars.p_compliant .> 0.5f0)
                        n = length(y_all)

                        dm = XGBoost.DMatrix(X)
                        XGBoost.setinfo!(dm, "label", y_all)

                        ρ̃ = fill(Ρ̃_INIT, n)
                        saved_p_bytes = UInt8[]
                        saved_r_bytes = UInt8[]
                        last_sweep = 0

                        for sweep in 1:MAX_INNER_SWEEPS
                            last_sweep = sweep
                            ρ̃_prev = copy(ρ̃)

                            booster_p = Supply.fit_xgb_custom(
                                dm; num_round = NUM_ROUND_STEP1,
                                params = PBASE_XGB_PARAMS,
                                xgb_model = saved_p_bytes,
                            ) do ŷ
                                grad = Vector{Float32}(undef, n)
                                hess = Vector{Float32}(undef, n)
                                @inbounds for j in 1:n
                                    grad[j], hess[j] = Supply.pbase_grad_hess(
                                        ŷ[j], y_all[j], γ_all[j], ρ̃[j]
                                    )
                                end
                                return grad, hess
                            end
                            p = logistic.(XGBoost.predict(booster_p, dm; margin = true))
                            saved_p_bytes = copy(
                                XGBoost.save(booster_p, Vector{UInt8}; format = "ubj")
                            )
                            finalize(booster_p)

                            complier_local = findall(==(1.0f0), γ_all)
                            if isempty(complier_local)
                                saved_r_bytes = UInt8[]
                                fill!(ρ̃, Supply.RHO_MIN)
                            else
                                X_c = X[complier_local, :]
                                y_c = y_all[complier_local]
                                p_c = p[complier_local]
                                n_c = length(complier_local)
                                dm_c = XGBoost.DMatrix(X_c)
                                XGBoost.setinfo!(dm_c, "label", y_c)

                                booster_r = Supply.fit_xgb_custom(
                                    dm_c;
                                    num_round = NUM_ROUND_STEP1,
                                    params = RHO_XGB_PARAMS,
                                    xgb_model = saved_r_bytes,
                                ) do ŷ
                                    grad = Vector{Float32}(undef, n_c)
                                    hess = Vector{Float32}(undef, n_c)
                                    @inbounds for j in 1:n_c
                                        grad[j], hess[j] = Supply.rho_grad_hess(
                                            ŷ[j], y_c[j], 1.0f0, p_c[j]
                                        )
                                    end
                                    return grad, hess
                                end
                                fill!(ρ̃, Supply.RHO_MIN)
                                ρ̃[complier_local] .= Supply.rho_from_margin.(
                                    XGBoost.predict(booster_r, dm_c; margin = true)
                                )
                                saved_r_bytes = copy(
                                    XGBoost.save(booster_r, Vector{UInt8}; format = "ubj")
                                )
                                finalize(booster_r)
                                finalize(dm_c)
                            end

                            maximum(abs, ρ̃ .- ρ̃_prev) < Ρ̃_TOL && break
                        end

                        finalize(dm)
                        (; model, model_year, extra) = cars[1, :]
                        @info "  refit" model model_year extra last_sweep n_compliers = count(
                            ==(1.0f0), γ_all
                        ) n_total = n
                        return (model, model_year, extra) => (saved_p_bytes, saved_r_bytes)
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
            JLD2.save_object("data/compliance_fits_final_stratified.jld2", final_M_step)
            return nothing
        end
    end
    return nothing
end
