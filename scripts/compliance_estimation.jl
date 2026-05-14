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
    using Memoize
    import Folds
    using FoldsThreads

    Φ = cdf$(Normal())

    DEMOS = Empirics.DEFAULT_DEMOGRAPHICS

    include("plot_assortment_probs.jl")

    # calculates the index of the closest dealers, and standardizes demographics
    function prepare_dealers!(dealer_demographics)
        leftjoin!(dealer_demographics, select(Dealers.State.df, :uid, :geometry), on = :uid)
        sort!(dealer_demographics, :uid)
        D =
            Iterators.product(dealer_demographics.geometry, dealer_demographics.geometry) .|>
            splat(Geometry.distance) .|>
            Geometry.miles
        dealer_demographics.closest_others = map(d -> Tuple(sortperm(d)[2:4]), eachrow(D))
        M = Matrix(dealer_demographics[!, DEMOS])
        M .-= mean(M, dims = 1)
        M *= sqrt(inv(cov(M)))
        dealer_demographics[!, DEMOS] .= M
        return nothing
    end

    @memoize function X_matrix(compliers, dealer_demographics, compliance)
        X = mapreduce(vcat, eachrow(compliers)) do c
            others_D = Matrix(dealer_demographics[collect(c.closest_others), DEMOS])
            my_D = Vector(pluck(dealer_demographics, :uid => ByRow(==(c.dealer_uid)))[DEMOS])
            others_comply = map(c.closest_others) do i
                try
                    pluck(
                        compliance,
                        :dealer_uid => ByRow(==(dealer_demographics.uid[i])),
                        :t => ByRow(==(c.t)),
                    ).is_compliant
                catch
                    false
                end
            end
            return vcat(
                Dates.year(c.t),
                Dates.month(c.t),
                my_D,
                collect(others_comply),
                reshape(others_D, :),
            )'
        end
        return FD.value.(X)
    end

    xgboost_params = (
        max_depth = 3,
        num_round = 10,
        eta = 0.15,
        tree_method = "hist",
        subsample = 0.8,
        colsample_bytree = 0.8,
        min_child_weight = 15,
        lambda = 10.0,
        max_bin = 32,
    )

    # calculate the log likelihood, concentrating out rho tilde
    function loglikelihood(α, params)
        (; cars, dealer_demographics, compliance) = params
        α_intercept, α_demos... = α
        base_probs = DataFrame(
            dealer_uid = dealer_demographics.uid,
            p_base = Φ.(α_intercept .+ Matrix(dealer_demographics[!, DEMOS]) * α_demos),
            closest_others = dealer_demographics.closest_others,
        )
        cars = leftjoin(cars, base_probs, on = :dealer_uid)
        compliers = subset(cars, :is_compliant)
        ρ = @. FD.value((compliers.is_sold - compliers.p_base) / (1 - compliers.p_base))
        X = X_matrix(compliers, dealer_demographics, compliance)
        ρ_hat = with_logger(NullLogger()) do
            XGBoost.predict(xgboost((X, ρ); xgboost_params...), X)
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

    function custom_bhhh(cars, dealer_demographics, compliance; max_iter = 20)
        (; model, model_year, extra) = cars[1, :]
        tag = "$model_year $model $extra"
        F_i = Base.Fix2((-) ∘ loglikelihood, (; cars, dealer_demographics, compliance))
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
        α[1] = quantile(Normal(), mean(cars.is_sold))
        a_init = 1.0
        for i in 1:max_iter
            g = G(α)
            if norm(g) < 1e-3
                @info "[$tag] Gradient convergence achieved, exiting"
                return α
            end
            if i % 1 == 0
                @info "[$tag] Progress $i" F(α) norm(g)
            end

            newton_step = H(α) \ g

            # line search
            ϕ(a) = F(α - a * newton_step)
            dϕ(a) = dot(-newton_step, G(α - a * newton_step))
            ϕdϕ(a) = (ϕ(a), dϕ(a))
            linesearch() =
                while true
                    try
                        return ls(ϕ, dϕ, ϕdϕ, a_init, F(α), dot(-g, newton_step))
                    catch e
                        if e isa XGBoost.XGBoostError
                            @warn "[$tag] XGBoost error at $(a_init)"
                            a_init /= 2
                        else
                            rethrow(e)
                        end
                    end
                end
            a_opt, _ = linesearch()
            if norm(a_opt * newton_step) < 1e-5
                @warn "[$tag] Not moving any more; exiting"
                return α
            end
            α -= a_opt * newton_step
        end
        @warn "Uncoverged after 10 steps"
        return α
    end
end

function (@main)(_)
    Dealers.load_from_disk!(snakemake.input["dealer_init"])
    fords = read_arrow(snakemake.input["fords"])
    dealer_demographics = semijoin(
        read_arrow(snakemake.input["dealer_demographics"]), fords, on = :uid => :dealer_uid
    )
    prepare_dealers!(dealer_demographics)

    panel = make_panel(fords, dealer_demographics)
    select!(panel, :dealer_uid, :t, :model, :model_year, :extra, :is_sold)

    compliance = selectunique(panel, [:dealer_uid, :t])
    compliance.is_compliant = rand(nrow(compliance)) .> 0.5

    gdf = groupby(panel, [:model, :model_year, :extra])

    for i in 1:100
        @info "Starting round $i"
        M_step =
            let mapped = pmap(collect(gdf)) do cars
                    cars = leftjoin(cars, compliance, on = [:dealer_uid, :t])
                    coefs = custom_bhhh(cars, dealer_demographics, compliance)
                    α_intercept, α_demos... = coefs
                    base_probs = DataFrame(
                        dealer_uid = dealer_demographics.uid,
                        p_base = Φ.(α_intercept .+ Matrix(dealer_demographics[!, DEMOS]) * α_demos),
                        closest_others = dealer_demographics.closest_others,
                    )
                    cars = leftjoin(cars, base_probs, on = :dealer_uid)
                    compliers = subset(cars, :is_compliant)
                    ρ = @. FD.value((compliers.is_sold - compliers.p_base) / (1 - compliers.p_base))
                    X = X_matrix(compliers, dealer_demographics, compliance)
                    booster = with_logger(NullLogger()) do
                        xgboost((X, ρ); xgboost_params...)
                    end
                    (; model, model_year, extra) = cars[1, :]
                    return (model, model_year, extra) =>
                        (Tuple(coefs), XGBoost.serialize(booster))
                end
                mapreduce(
                    Dict,
                    merge,
                    mapped,
                    init = Dict{
                        Tuple{String, Int64, String},
                        Tuple{NTuple{7, Float64}, Vector{UInt8}},
                    }(),
                )
            end

        compliance_new = copy(compliance)
    end
end
