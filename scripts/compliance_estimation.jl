using Distributed
@everywhere using Memoize

@everywhere begin
    using DealerMnA
    using XGBoost
    using LoggingExtras
    using LinearAlgebra
    using LogExpFunctions

    BLAS.set_num_threads(1)

    DEMOS = Empirics.DEFAULT_DEMOGRAPHICS

    # K-fold cross-fitting for honest p_base and ρ̃; both share the fold map.
    K = 3

    # Alternating fits of (p_base, ρ̃) within each M-step fold.
    N_INNER_SWEEPS = 2

    include("plot_assortment_probs.jl")

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

    xgboost_params = (
        # Identity link in XGBoost's view; our custom objective re-interprets
        # raw output as logit-space and applies σ in the gradient/Hessian.
        objective = "reg:squarederror",
        # Raw-space 0 ⇒ σ(0) = 0.5 starting probability for both p_base and ρ̃.
        # XGBoost's "reg:squarederror" default is 0.5 in raw space (σ(0.5) ≈ 0.62),
        # which biases predictions up and makes ρ̃ "look positive" even on random
        # partitions, defeating CEM.
        base_score = 0.0,
        max_depth = 3,
        num_round = 4,
        lambda = 10.0,
        eta = 0.15,
        tree_method = "hist",
        subsample = 0.8,
        min_child_weight = 25,
        max_bin = 32,
        seed = 85704,
    )
    XGB_NUM_ROUND = xgboost_params.num_round
    XGB_BOOSTER_PARAMS = Base.structdiff(xgboost_params, (num_round = nothing,))

    # ---- Custom mixture-likelihood objectives for XGBoost ----
    #
    # Both p_base and ρ̃ are σ(raw XGBoost output) under a logistic link. Per-row
    # gradient and Fisher-information Hessian derived from the mixture data
    # log-likelihood; Hessian is positive for all p, ρ̃, y. See plan file for
    # the full derivation.

    # p_base loss derivative w.r.t. raw output f, given y, complier flag z,
    # and current ρ̃ estimate (held fixed).
    function pbase_grad_hess(f::Real, y::Real, z::Real, ρ̃::Real)
        p = logistic(f)
        if iszero(z)
            return Float32(p - y), Float32(p * (1 - p))
        end
        u = 1 - ρ̃
        q = p + (1 - p) * ρ̃
        g = (1 - y) * p - y * u * p * (1 - p) / q
        h = u * p^2 * (1 - p) / q
        return Float32(g), Float32(h)
    end

    # ρ̃ loss derivative w.r.t. raw output gᵣ, given y and current p_base
    # (held fixed). Only applied to complier-classified rows.
    function rho_grad_hess(gᵣ::Real, y::Real, p::Real)
        ρ̃ = logistic(gᵣ)
        q = p + (1 - p) * ρ̃
        g = (1 - y) * ρ̃ - y * (1 - p) * ρ̃ * (1 - ρ̃) / q
        h = (1 - p) * ρ̃^2 * (1 - ρ̃) / q
        return Float32(g), Float32(h)
    end

    # Train a Booster for XGB_NUM_ROUND iterations using a custom per-row
    # gradient/Hessian. compute_g_h(ŷ::Vector) -> (grad::Vector, hess::Vector)
    # where ŷ is the raw (margin) prediction at each row of dm. Closure-first
    # signature so callers can use `do`-block syntax.
    function fit_xgb_custom(compute_g_h, dm::XGBoost.DMatrix)
        b = with_logger(NullLogger()) do
            XGBoost.Booster(dm; XGB_BOOSTER_PARAMS...)
        end
        for _ in 1:XGB_NUM_ROUND
            ŷ = XGBoost.predict(b, dm; margin = true)
            grad, hess = compute_g_h(ŷ)
            with_logger(NullLogger()) do
                XGBoost.updateone!(b, dm, grad, hess)
            end
        end
        return b
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
        @info "  γ = p_compliant" mean = mean(p_compliant) q = quantile(p_compliant, qs) frac_above_0_9 =
            mean(>(0.9), p_compliant) frac_below_0_1 = mean(<(0.1), p_compliant)
        @info "  ρ̃ (mean per d,t)" mean = mean(ρ̃_means) q = quantile(ρ̃_means, qs)
        @info "  p_base (mean per d,t)" mean = mean(p_base_means) q = quantile(p_base_means, qs)
        @info "  cars per (d,t)" mean = mean(n_cars_vec) q = quantile(n_cars_vec, qs)
        @info "  within-(d,t) variance" obs = mean(obs_vars) nc_pred = mean(nc_var_pred) frac_obs_above_nc =
            mean(obs_vars .> nc_var_pred)
        @info "  observed sale rate vs p_base" obs = mean(obs_means) p_base = mean(p_base_means) lift_mean =
            mean(sale_lift) lift_q = quantile(sale_lift, qs)
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

    panel = make_panel(fords, dealer_demographics)
    select!(panel, :dealer_uid, :t, :model, :model_year, :extra, :is_sold)
    subset!(panel, :t => ByRow(≥(Date(2017))))

    compliance = selectunique(panel, [:dealer_uid, :t])
    compliance.p_compliant = rand(nrow(compliance))

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
        compliance.is_compliant = compliance.p_compliant .> 0.5
        π = mean(compliance.is_compliant)
        M_step =
            let mapped = pmap(collect(gdf)) do cars
                    cars = leftjoin(cars, compliance, on = [:dealer_uid, :t])
                    X = X_matrix(cars)
                    y_all = Float32.(cars.is_sold)
                    z_all = Float32.(cars.is_compliant)
                    folds = [fold_map[(c.dealer_uid, c.t)] for c in eachrow(cars)]

                    pbase_bytes = Vector{Vector{UInt8}}(undef, K)
                    rho_bytes = Vector{Vector{UInt8}}(undef, K)

                    for k in 1:K
                        train_idx = findall(!=(k), folds)
                        X_tr = X[train_idx, :]
                        y_tr = y_all[train_idx]
                        z_tr = z_all[train_idx]
                        complier_local = findall(==(1.0f0), z_tr)
                        n_tr = length(train_idx)

                        dm_tr = XGBoost.DMatrix(X_tr)
                        XGBoost.setinfo!(dm_tr, "label", y_tr)

                        ρ̃_tr = zeros(Float32, n_tr)
                        saved_p_bytes = UInt8[]
                        saved_r_bytes = UInt8[]

                        for sweep in 1:N_INNER_SWEEPS
                            # p_base fit: every row contributes
                            booster_p = fit_xgb_custom(dm_tr) do ŷ
                                grad = Vector{Float32}(undef, n_tr)
                                hess = Vector{Float32}(undef, n_tr)
                                @inbounds for j in 1:n_tr
                                    grad[j], hess[j] = pbase_grad_hess(
                                        ŷ[j], y_tr[j], z_tr[j], ρ̃_tr[j]
                                    )
                                end
                                return grad, hess
                            end
                            p_tr =
                                logistic.(XGBoost.predict(booster_p, dm_tr; margin = true))
                            sweep == N_INNER_SWEEPS &&
                                (saved_p_bytes = copy(XGBoost.serialize(booster_p)))
                            finalize(booster_p)

                            isempty(complier_local) && continue

                            # ρ̃ fit: complier-classified rows only, p_base held fixed
                            X_tr_c = X_tr[complier_local, :]
                            y_tr_c = y_tr[complier_local]
                            p_tr_c = p_tr[complier_local]
                            n_tr_c = length(complier_local)
                            dm_tr_c = XGBoost.DMatrix(X_tr_c)
                            XGBoost.setinfo!(dm_tr_c, "label", y_tr_c)
                            booster_r = fit_xgb_custom(dm_tr_c) do ŷ
                                grad = Vector{Float32}(undef, n_tr_c)
                                hess = Vector{Float32}(undef, n_tr_c)
                                @inbounds for j in 1:n_tr_c
                                    grad[j], hess[j] = rho_grad_hess(
                                        ŷ[j], y_tr_c[j], p_tr_c[j]
                                    )
                                end
                                return grad, hess
                            end
                            ρ̃_tr[complier_local] .=
                                logistic.(XGBoost.predict(booster_r, dm_tr_c; margin = true))
                            sweep == N_INNER_SWEEPS &&
                                (saved_r_bytes = copy(XGBoost.serialize(booster_r)))
                            finalize(booster_r)
                            finalize(dm_tr_c)
                        end

                        pbase_bytes[k] = saved_p_bytes
                        rho_bytes[k] = saved_r_bytes
                        finalize(dm_tr)
                    end

                    (; model, model_year, extra) = cars[1, :]
                    return (model, model_year, extra) => (pbase_bytes, rho_bytes)
                end
                mapreduce(
                    Dict,
                    merge,
                    mapped,
                    init = Dict{
                        Tuple{String, Int64, String},
                        Tuple{Vector{Vector{UInt8}}, Vector{Vector{UInt8}}},
                    }(),
                )
            end

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
                XGBoost.deserialize!(booster, pbase_bytes)
                f_raw = only(XGBoost.predict(booster, dm; margin = true))
                clamp(logistic(f_raw), 1e-12, 1 - 1e-12)
            end

            cars.ρ_tilde = map(eachrow(cars)) do (; model, model_year, extra)
                rho_bytes = M_step[(model, model_year, extra)][2][k]
                # Empty bytes ⇒ this fold had no compliers in training; fall
                # back to ρ̃ ≈ 0 so the complier branch collapses to noncomplier.
                isempty(rho_bytes) && return 1e-12
                XGBoost.deserialize!(booster, rho_bytes)
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
            p_compliant = softmax([log(π) + ll_complier, log(1 - π) + ll_noncomplier])[1]
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

        γ = 0.8
        next_p_compliant = γ .* compliance.p_compliant .+ (1 - γ) .* E_step.p_compliant
        n_flipped = sum((next_p_compliant .> 0.5) .!= compliance.is_compliant)
        step_change = norm(E_step.p_compliant - compliance.p_compliant)
        max_change = maximum(abs, E_step.p_compliant - compliance.p_compliant)
        @info "Round $i" n_flipped step_change max_change π mean_p_compliant = mean(E_step.p_compliant)

        if n_flipped == 0
            @info "Partition stable, CEM converged at step $i"
            Arrow.write("data/compliance.arrow", E_step)
            JLD2.save_object("data/compliance_fits.jld2", M_step)
            return nothing
        end
        compliance.p_compliant .= next_p_compliant
    end
    @error "Boohoo, no convergence after 100 steps"
    return nothing
end
