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

    Φ = cdf$(Normal())

    DEMOS = Empirics.DEFAULT_DEMOGRAPHICS

    include("plot_assortment_probs.jl")

    # calculates the index of the closest dealers, and standardizes demographics
    function prepare_dealers!(dealer_demographics)
        M = Matrix(dealer_demographics[!, DEMOS])
        M .-= mean(M, dims = 1)
        M *= sqrt(inv(cov(M)))
        dealer_demographics[!, DEMOS] .= M # demos now approx a multivariate standard normal
        return nothing
    end

    # tf_t_j has a consistent ordering, so enumerate the indexes for fast lookup
    @memoize function state_lookup(model)
        intermed = Supply.tf_to_j_lookup(model)
        return Dict(k => i for (i, k) in enumerate(keys(intermed)))
    end

    @memoize function X_matrix(compliers, model, state_matrix)
        X = mapreduce(vcat, eachrow(compliers)) do c
            quarter = (Dates.year(c.t), Dates.quarterofyear(c.t))
            row_idx = state_lookup(model)[(quarter, c.dealer_uid)]
            return reshape(state_matrix[row_idx, :], 1, :)
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
        seed = 85704,
    )

    # calculate the log likelihood, concentrating out rho tilde
    function loglikelihood(α, params)
        (; cars, dealer_demographics, state_matrix, demand_model) = params
        α_intercept, α_demos... = α
        base_probs = DataFrame(
            dealer_uid = dealer_demographics.uid,
            p_base = Φ.(α_intercept .+ Matrix(dealer_demographics[!, DEMOS]) * α_demos),
        )
        cars = leftjoin(cars, base_probs, on = :dealer_uid)
        compliers = subset(cars, :is_compliant)
        ρ = @. FD.value((compliers.is_sold - compliers.p_base) / (1 - compliers.p_base))
        X = X_matrix(compliers, demand_model, state_matrix)
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

    function custom_bhhh(params; max_iter = 20)
        (; model, model_year, extra) = params.cars[1, :]
        tag = "$model_year $model $extra"
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
        @warn "Uncoverged after $max_iter steps"
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
    subset!(panel, :t => ByRow(≥(Date(2017))))

    demand_model = Demand.DemandModel(; JLD2.load_object(snakemake.input["model_frames"])...)

    state_matrix = JLD2.load_object(snakemake.input["state_matrix"])

    compliance = selectunique(panel, [:dealer_uid, :t])
    compliance.p_compliant = rand(nrow(compliance))

    gdf = groupby(panel, [:model, :model_year, :extra])

    M_step = Dict()
    E_step = empty!(copy(compliance))

    for i in 1:100
        @info "Starting round $i"
        compliance.is_compliant = compliance.p_compliant .> 0.5
        M_step =
            let mapped = pmap(collect(gdf)) do cars
                    cars = leftjoin(cars, compliance, on = [:dealer_uid, :t])
                    coefs = custom_bhhh(
                        (; cars, dealer_demographics, demand_model, state_matrix),
                        max_iter = 20,
                    )
                    α_intercept, α_demos... = coefs
                    base_probs = DataFrame(
                        dealer_uid = dealer_demographics.uid,
                        p_base = Φ.(α_intercept .+ Matrix(dealer_demographics[!, DEMOS]) * α_demos),
                    )
                    cars = leftjoin(cars, base_probs, on = :dealer_uid)
                    compliers = subset(cars, :is_compliant)
                    ρ = @. FD.value((compliers.is_sold - compliers.p_base) / (1 - compliers.p_base))
                    X = X_matrix(compliers, demand_model, state_matrix)
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

        E_step = copy(compliance)
        E_step.p_compliant = pmap(eachrow(E_step)) do r
            cars = subset(panel, :dealer_uid => ByRow(==(r.dealer_uid)), :t => ByRow(==(r.t)))
            demos = pluck(dealer_demographics, :uid => ByRow(==(r.dealer_uid)))
            cars.p_base = map(eachrow(cars)) do (; model, model_year, extra)
                α_intercept, α_demo... = M_step[(model, model_year, extra)][1]
                Φ(α_intercept + dot(α_demo, demos[DEMOS]))
            end
            X = X_matrix(hcat(cars[[1], :], DataFrame(demos)), demand_model, state_matrix)
            cars.ρ_tilde = map(eachrow(cars)) do (; model, model_year, extra)
                booster = XGBoost.Booster(X)
                XGBoost.deserialize!(booster, M_step[(model, model_year, extra)][2])
                clamp(only(XGBoost.predict(booster, X)), 1e-12, 1 - 1e-12)
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
            return softmax([ll_complier, ll_noncomplier])[1]
        end

        step_change = norm(E_step.p_compliant - compliance.p_compliant)
        @info "Change: $(round(step_change, digits = 6))"

        if step_change < 1e-4
            @info "Wooohoo! We converged at step $i"
            Arrow.write("data/compliance.arrow", E_step)
            JLD2.save_object("data/compliance_fits.jld2", M_step)
            break
        end
        compliance.p_compliant .= E_step.p_compliant
    end
    @error "Boohoo, no convergence after 100 steps"
    return nothing
end
