using Distributed
@everywhere using Memoize
@everywhere using PartialFunctions

@everywhere begin
    using DealerMnA
    using Distributions
    using XGBoost
    import ForwardDiff as FD
    using LoggingExtras
    using LineSearches
    using LinearAlgebra
    using LogExpFunctions

    BLAS.set_num_threads(1)

    Φ = cdf$(Normal())

    DEMOS = Empirics.DEFAULT_DEMOGRAPHICS

    # cross-fit folds for honest ρ̃ estimation
    K = 3

    include("plot_assortment_probs.jl")

    demand_model = Demand.DemandModel(; JLD2.load_object("data/model_frames.jld2")...)

    state_matrix = JLD2.load_object("data/residual_demand.jld2")

    # calculates the index of the closest dealers, and standardizes demographics
    function prepare_dealers!(dealer_demographics)
        M = Matrix(dealer_demographics[!, DEMOS])
        M .-= mean(M, dims = 1)
        M *= sqrt(inv(cov(M)))
        dealer_demographics[!, DEMOS] .= M # demos now approx a multivariate standard normal
        return nothing
    end

    # tf_t_j has a consistent ordering, so enumerate the indexes for fast lookup
    @memoize function state_lookup()
        intermed = Supply.tf_to_j_lookup(demand_model)
        return Dict(k => i for (i, k) in enumerate(keys(intermed)))
    end

    @memoize function X_matrix(compliers)
        nrows = nrow(compliers)
        ncols = size(state_matrix, 2)
        X = Matrix{Float32}(undef, nrows, ncols)
        lookup = state_lookup()
        for (i, c) in enumerate(eachrow(compliers))
            quarter = (Dates.year(c.t), Dates.quarterofyear(c.t))
            X[i, :] .= @view state_matrix[lookup[(quarter, c.dealer_uid)], :]
        end
        return X
    end

    xgboost_params = (
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

    # calculate the log likelihood, concentrating out rho tilde via K-fold
    # cross-fitted XGBoost: each complier's ρ̃ comes from a booster trained on
    # rows whose fold differs from its own.
    function loglikelihood(α, params)
        (; cars, dealer_demographics, train_dms, test_dms, train_idx, test_idx) =
            params
        α_intercept, α_demos... = α
        base_probs = DataFrame(
            dealer_uid = dealer_demographics.uid,
            p_base = Φ.(α_intercept .+ Matrix(dealer_demographics[!, DEMOS]) * α_demos),
        )
        cars = leftjoin(cars, base_probs, on = :dealer_uid)
        compliers = subset(cars, :is_compliant)
        ρ = @. FD.value((compliers.is_sold - compliers.p_base) / (1 - compliers.p_base))

        ρ_hat = Vector{Float32}(undef, length(ρ))
        for k in 1:K
            XGBoost.setlabel!(train_dms[k], Float32.(ρ[train_idx[k]]))
            booster_k = with_logger(NullLogger()) do
                xgboost(train_dms[k]; xgboost_params...)
            end
            ρ_hat[test_idx[k]] = XGBoost.predict(booster_k, test_dms[k])
        end
        compliers.ρ_tilde = clamp.(ρ_hat, 1e-12, 1 - 1e-12)

        ll_compliers = let (; is_sold, p_base, ρ_tilde) = compliers
            @. is_sold * log(p_base + (1 - p_base) * ρ_tilde) +
                (1 - is_sold) * log((1 - p_base) * (1 - ρ_tilde))
        end

        noncompliers = subset(cars, :is_compliant => ByRow(!))
        ll_noncompliers = let (; is_sold, p_base) = noncompliers
            @. is_sold * log(p_base) + (1 - is_sold) * log(1 - p_base)
        end

        return vcat(ll_compliers, ll_noncompliers)
    end

    α_0 = zeros(length(DEMOS) + 1)
    ls = BackTracking()

    function custom_bhhh(params; max_iter = 50)
        (; model, model_year, extra) = params.cars[1, :]
        tag = "$model_year $model $extra"
        compliers = subset(params.cars, :is_compliant)
        X = X_matrix(compliers)
        folds = [params.fold_map[(c.dealer_uid, c.t)] for c in eachrow(compliers)]
        train_idx = [findall(!=(k), folds) for k in 1:K]
        test_idx = [findall(==(k), folds) for k in 1:K]
        train_dms = [XGBoost.DMatrix(X[train_idx[k], :]) for k in 1:K]
        test_dms = [XGBoost.DMatrix(X[test_idx[k], :]) for k in 1:K]
        params = (; params..., train_dms, test_dms, train_idx, test_idx)
        F_i = Base.Fix2((-) ∘ loglikelihood, params)
        F = sum ∘ F_i
        G_i = let gcache = FD.JacobianConfig(F_i, α_0)
            @$ FD.jacobian(F_i, _, gcache)
        end
        G = (@$ reshape(_, :)) ∘ (@$ sum(_; dims = 1)) ∘ G_i
        H(α) =
            let g = G_i(α)
                g' * g
            end
        α = copy(α_0)
        α[1] = quantile(Normal(), mean(params.cars.is_sold))
        a_init = 1.0
        for i in 1:max_iter
            g = G(α)
            if norm(g) < 1e-3
                @info "[$tag] Gradient convergence achieved, exiting"
                return α
            end
            if i % 10 == 0
                @info "[$tag] Progress $i" F(α) norm(g)
            end

            # if it's singular, just use gradient direction
            newton_step = try
                H(α) \ g
            catch e
                if e isa SingularException
                    @warn "[$tag] singular exception"
                    g
                else
                    rethrow(e)
                end
            end

            # line search
            ϕ(a) = F(α - a * newton_step)
            dϕ(a) = dot(-newton_step, G(α - a * newton_step))
            ϕdϕ(a) = (ϕ(a), dϕ(a))
            linesearch() =
                let n = 100
                    while true
                        iszero(n) && return (0, 0)
                        try
                            return ls(ϕ, dϕ, ϕdϕ, a_init, F(α), dot(-g, newton_step))
                        catch e
                            if e isa XGBoost.XGBoostError
                                n -= 1
                                a_init /= 1.2
                            else
                                rethrow(e)
                            end
                        end
                    end
                end
            a_opt, _ = linesearch()
            if norm(a_opt * newton_step) < 1e-6
                @warn "[$tag] Not moving any more; exiting"
                return α
            end
            α -= a_opt * newton_step
        end
        @warn "[$tag] Uncoverged after $max_iter steps"
        return α
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
    # every vehicle's M-step booster and matches its E-step prediction.
    fold_map = Dict(
        (row.dealer_uid, row.t) => mod(hash((row.dealer_uid, row.t)), K) + 1 for
        row in eachrow(compliance)
    )

    gdf = groupby(panel, [:model, :model_year, :extra])

    M_step = Dict()
    E_step = empty!(copy(compliance))

    for i in 1:100
        @info "Starting round $i"
        compliance.is_compliant = compliance.p_compliant .> 0.5
        π = mean(compliance.is_compliant)
        M_step =
            let mapped = pmap(collect(gdf)) do cars
                    empty!(Memoize.memoize_cache(X_matrix))
                    cars = leftjoin(cars, compliance, on = [:dealer_uid, :t])
                    coefs = custom_bhhh(
                        (; cars, dealer_demographics, fold_map),
                        max_iter = 50,
                    )
                    α_intercept, α_demos... = coefs
                    base_probs = DataFrame(
                        dealer_uid = dealer_demographics.uid,
                        p_base = Φ.(α_intercept .+ Matrix(dealer_demographics[!, DEMOS]) * α_demos),
                    )
                    cars = leftjoin(cars, base_probs, on = :dealer_uid)
                    compliers = subset(cars, :is_compliant)
                    ρ = @. FD.value((compliers.is_sold - compliers.p_base) / (1 - compliers.p_base))
                    X = X_matrix(compliers)
                    folds = [fold_map[(c.dealer_uid, c.t)] for c in eachrow(compliers)]
                    boosters_bytes = map(1:K) do k
                        train_idx_k = findall(!=(k), folds)
                        dm_k = XGBoost.DMatrix(X[train_idx_k, :])
                        XGBoost.setinfo!(dm_k, "label", Float32.(ρ[train_idx_k]))
                        booster_k = with_logger(NullLogger()) do
                            xgboost(dm_k; xgboost_params...)
                        end
                        # copy: XGBoost.serialize returns a Vector{UInt8} backed
                        # by C-owned memory tied to booster_k's lifetime; copy
                        # before the booster goes out of scope.
                        copy(XGBoost.serialize(booster_k))
                    end
                    (; model, model_year, extra) = cars[1, :]
                    return (model, model_year, extra) =>
                        (Tuple(coefs), boosters_bytes)
                end
                mapreduce(
                    Dict,
                    merge,
                    mapped,
                    init = Dict{
                        Tuple{String, Int64, String},
                        Tuple{NTuple{7, Float64}, Vector{Vector{UInt8}}},
                    }(),
                )
            end

        @info "Starting E step"

        E_step = copy(compliance)
        E_results = pmap(eachrow(E_step); batch_size = 100) do r
            cars = subset(panel, :dealer_uid => ByRow(==(r.dealer_uid)), :t => ByRow(==(r.t)))
            demos = pluck(dealer_demographics, :uid => ByRow(==(r.dealer_uid)))
            cars.p_base = map(eachrow(cars)) do (; model, model_year, extra)
                α_intercept, α_demo... = M_step[(model, model_year, extra)][1]
                Φ(α_intercept + dot(α_demo, demos[DEMOS]))
            end
            k = fold_map[(r.dealer_uid, r.t)]
            dm = XGBoost.DMatrix(X_matrix(hcat(cars[[1], :], DataFrame(demos))))
            booster = XGBoost.Booster(dm)
            cars.ρ_tilde = map(eachrow(cars)) do (; model, model_year, extra)
                XGBoost.deserialize!(booster, M_step[(model, model_year, extra)][2][k])
                clamp(only(XGBoost.predict(booster, dm)), 1e-12, 1 - 1e-12)
            end
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
            )
        end
        E_step.p_compliant = [r.p_compliant for r in E_results]
        ll_gap = [r.ll_complier - r.ll_noncomplier for r in E_results]
        ρ̃_means = [r.ρ̃_mean for r in E_results]
        p_base_means = [r.p_base_mean for r in E_results]
        n_cars_vec = [r.n_cars for r in E_results]
        qs = [0.1, 0.5, 0.9]
        @info "  ll_complier - ll_noncomplier per (d,t)" mean = mean(ll_gap) q = quantile(
            ll_gap, qs
        ) n_positive = sum(>(0), ll_gap) total = length(ll_gap)
        @info "  γ = p_compliant" mean = mean(E_step.p_compliant) q = quantile(
            E_step.p_compliant, qs
        ) frac_above_0_9 = mean(>(0.9), E_step.p_compliant) frac_below_0_1 = mean(
            <(0.1), E_step.p_compliant
        )
        @info "  ρ̃ (mean per d,t)" mean = mean(ρ̃_means) q = quantile(ρ̃_means, qs)
        @info "  p_base (mean per d,t)" mean = mean(p_base_means) q = quantile(p_base_means, qs)
        @info "  cars per (d,t)" mean = mean(n_cars_vec) q = quantile(n_cars_vec, qs)

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
            break
        end
        compliance.p_compliant .= next_p_compliant
    end
    @error "Boohoo, no convergence after 100 steps"
    return nothing
end
