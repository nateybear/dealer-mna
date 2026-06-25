using DealerMnA
using Plots
using StatsPlots
using StatsBase: cov, ordinalrank
using Distributions
using LinearAlgebra
using OhMyThreads
using FixedEffectModels
using Printf

import Random
Random.seed!(85704)

D = vcat(Empirics.DEFAULT_DEMOGRAPHICS, [:closest_first, :closest_second])

selectunique(X, cols) = select(unique(X, cols), cols)

struct Mahalanobis{T}
    Σ::T
end
mahalanobis(X) = Mahalanobis(qr(cov(Matrix(X))))
(d::Mahalanobis)(x) = sqrt(only(x' * (d.Σ \ x)))
Base.Broadcast.broadcastable(d::Mahalanobis) = Ref(d)

function optimal_bandwidth(X, y; distance = mahalanobis(X), K = pdf$Normal(), binary = false)
    h_steps = 10 .^ range(-0.5, 2.5, length = 25)
    train_idxs = sample(axes(X, 1), floor(Int64, 0.8 * Base.size(X, 1)), replace = false)
    sse = map(h_steps) do h_candidate
        ŷ = try
            npregress(
                X[train_idxs, :],
                y[train_idxs],
                X[Not(train_idxs), :];
                h = h_candidate,
                distance,
                K,
                binary,
            )[
                :, 1
            ]
        catch
            Inf
        end
        sum((y[Not(train_idxs)] .- ŷ) .^ 2)
    end
    @info "Optimal bandwidth: $(h_steps[argmin(sse)])"
    return h_steps[argmin(sse)]
end

logit(x) = @. log(x / (1 - x))
invlogit(x) = @. 1 / (1 + exp(-x))

function npregress(
    X, y, Xstar = X; h = nothing, distance = mahalanobis(X), K = pdf$Normal(), binary = false
)
    h = isnothing(h) ? optimal_bandwidth(X, y; distance, K, binary) : h
    if binary
        mapreduce(vcat, eachrow(Xstar)) do x0
            local μ = 0.5
            local vx = 0.0
            for i in 1:20
                η = logit(μ)
                @debug "Iteration $i" μ extrema(μ) extrema(η)
                z = @. η + (y - μ) / (μ * (1 - μ))
                new_K = let μ = μ
                    σ = @. μ * (1 - μ)
                    x -> begin
                        K(x) .* σ
                    end
                end
                z_new, vx = only(
                    eachrow(npregress(X, z, x0'; h, distance, binary = false, K = new_K))
                )
                μ_new = clamp(invlogit(z_new), 1e-6, 1 - 1e-6)
                if abs(μ - μ_new) < 1e-4
                    @debug "Early return at iteration $i"
                    return [μ_new vx * (exp(-z_new) / (1 + exp(-z_new))^2)^2]
                end
                μ = 0.5μ + 0.5μ_new
            end
            return [μ vx * (exp(-logit(μ)) / (1 + exp(-logit(μ)))^2)^2]
        end
    else
        return mapreduce(vcat, eachrow(Xstar)) do x
            Z = X .- x'
            W = X .- x' |>
                eachrow .|>
                distance .|>
                Base.Fix2(/, h) |>
                K |>
                Diagonal
            Z = [ones(Base.size(X, 1)) Z]
            βₓ = (Z' * W * Z) \ (Z' * W * y)
            u = y - Z * βₓ
            V_βₓ = inv(Z' * W * Z) * (Z' * W * Diagonal(u .^ 2) * W * Z) * inv(Z' * W * Z)
            return reduce(hcat, Iterators.flatten(zip(βₓ, diag(V_βₓ))))
        end
    end
end

lag(x) =
    let N = length(x)
        vcat(missing, first(x, N - 1))
    end

extract_stat((v, (k, col))) = k => col => v
extract_stats(stats) = extract_stat.(collect(pairs(stats)))

trim_cut(x) =
    if x < 1 / 3
        "Low"
    elseif x < 2 / 3
        "Mid"
    else
        "High"
    end

function make_panel(cars, dealers)
    out = @> cars begin
        # transform(:trim_rank => ByRow(trim_cut) => :trim_rank) 
        groupby([:t, :dealer_uid, :model, :model_year, :extra])
        combine(nrow => :q)
    end

    let appropriate_t = (m, t) -> (Date(m) - Quarter(1)) <= t <= (Date(m) + Year(1))
        subset!(out, [:model_year, :t] => ByRow(appropriate_t))
    end

    ## attach lagged aggregate statistics
    agg_statistics = @> cars groupby([:t, :dealer_uid]) combine(
        extract_stats(Empirics.STATISTICS)...
    )
    transform!(agg_statistics, :t => ByRow(Base.Fix2(-, Quarter(1))) => :t)
    DataFrames.rename!(Base.Fix1(*, "lag_"), agg_statistics, cols = Not([:t, :dealer_uid]))
    out = innerjoin(out, agg_statistics, on = [:t, :dealer_uid])

    out = tmapreduce(vcat, groupby(out, :t)) do sales
        J = @> sales selectunique([:model, :model_year, :extra])
        J.is_new_model_year = J.model_year .== maximum(J.model_year)
        F = @> sales selectunique(:dealer_uid)
        JF = crossjoin(J, F)
        JF.t .= first(sales.t)
        leftjoin!(
            JF,
            select(unique(sales, :dealer_uid), :dealer_uid, names(sales, r"lag")),
            on = :dealer_uid,
        ) # attach lagged sales
        leftjoin!(
            JF,
            select(sales, Not(names(sales, r"lag"))),
            on = [:t, :dealer_uid, :model, :model_year, :extra],
        )
        JF.q = coalesce.(JF.q, 0)
        JF.is_sold = JF.q .> 0
        leftjoin!(
            JF,
            select(dealers, :uid, D),
            on = :dealer_uid => :uid,
        )
        return JF
    end
    subset!(groupby(out, [:model, :model_year, :extra]), :t => (>(1) ∘ length ∘ unique))
    return out
end

# first residualize dealer fixed effects
# then, within period, residualize as a function of demographics
function calculate_ε!(panel)
    estimation_sample = selectunique(panel, [:dealer_uid, :t, :lag_total_units, D...])
    transform!(groupby(estimation_sample, :t)) do df
        X = Matrix(df[:, D])
        y = df.lag_total_units
        (; lag_ε = y - npregress(X, y)[:, 1])
    end
    leftjoin!(
        panel,
        select(estimation_sample, :dealer_uid, :t, :lag_ε),
        on = [:dealer_uid, :t],
    )
    return nothing
end

function (@main)(_)
    snakemake!()
    with_logging() do
        fords = read_arrow(snakemake.input["fords"])
        dealer_demographics = read_arrow(snakemake.input["dealer_demographics"])
        panel = make_panel(fords, dealer_demographics)
        # subset!(panel, :is_new_model_year)
        subset!(panel, :model_year => ByRow(<(2020))) # not a full year of data for 2020, so exclude
        calculate_ε!(panel)
        dropmissing!(panel, :lag_ε)
        transform!(groupby(panel, :dealer_uid), :lag_total_units => median => :avg_units)
        panel.lag_ε ./= panel.avg_units

        rankdense(x) =
            let r = ordinalrank(x)
                r ./ length(x)
            end

        panel = @> panel groupby(:t) transform(:lag_ε => rankdense => :lag_ε_rank)
        transform!(groupby(panel, [:model, :model_year, :extra, :t]), threads = false) do df
            X = Matrix(df[!, D])
            y = df.is_sold
            @info "Running" first(df.model) first(df.model_year) first(df.t) first(df.extra)
            (; is_sold_resid = y - npregress(X, y)[:, 1])
        end

        plots = let gdf = groupby(panel, [:model, :model_year, :extra])
            combine(gdf, threads = false) do df
                ε_range = range(quantile.(Ref(df.lag_ε_rank), [0.1, 0.9])..., length = 50)
                npmodel = npregress(
                    reshape(df.lag_ε_rank, :, 1),
                    df.is_sold_resid,
                    reshape(ε_range, :, 1),
                    h = 0.5,
                )
                p = plot(
                    ε_range,
                    npmodel[:, 1],
                    ribbon = 1.96 * sqrt.(npmodel[:, 2]),
                    xlabel = "Quantile Lagged ε",
                    ylabel = "Probability Sold",
                    label = """Model: $(first(df.model)), Year: $(first(df.model_year))
                    Trim: $(first(df.extra))
                    Avg derivative: $(@sprintf("%.2e", mean(npmodel[:, 3])))
                    Baseline probability: $(@sprintf("%.2f", mean(df.is_sold)))
                    """,
                )
                # sort!(out, :f_stat, rev = true)
                # range_out = out.ε |> extrema |> collect |> diff |> only
                # best_stat = out[1, :]
                # if abs(current_stat.t_stat) < 1.96
                #     break
                # end
                # slope = current_stat.npmodel[1, 3]
                # plot!(
                #     p
                # )
                # scatter!(p, df.lag_ε, df.is_sold_resid, label = "Data", alpha = 0.1, markerstrokewidth=0)
                return (;
                    p, is_sold_mean = mean(df.is_sold)
                )
            end
        end

        if !isdir(snakemake.output["plots"])
            mkdir(snakemake.output["plots"])
        end

        for (; model, model_year) in eachrow(selectunique(panel, [:model, :model_year]))
            performance_plots = subset(
                plots, [:model, :model_year] => ByRow((x...) -> x == (model, model_year))
            )
            dens = plot(xlabel = "Average Sales Probability", ylabel = "Density")
            for g in groupby(
                subset(panel, [:model, :model_year] => ByRow((x...) -> x == (model, model_year))),
                [:extra],
            )
                means = combine(groupby(g, :t), :is_sold => mean).is_sold_mean
                density!(dens, means, label = first(g.extra), xlims = (0, 1), bandwidth = 0.05)
            end
            out = plot(
                performance_plots.p..., dens, size = (1600, 900)
            )
            png(out, snakemake.output["plots"] * "/$(model_year)_$(model).png")
        end
        touch(snakemake.output["sentinel"])
    end
    return nothing
end
