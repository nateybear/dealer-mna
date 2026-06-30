#### Nonparametric (kernel) regression smoother.
####
#### Lifted from scripts/plot_assortment_probs.jl. Used to residualize dealer-
#### quarter statistics against demographics.

using LinearAlgebra
using Distributions: Normal, pdf
using StatsBase: cov, sample
using DataFrames: Not
using PartialFunctions: $

struct Mahalanobis{T}
    Σ::T
end

mahalanobis(X) = Mahalanobis(qr(cov(Matrix(X))))
(d::Mahalanobis)(x) = sqrt(only(x' * (d.Σ \ x)))
Base.Broadcast.broadcastable(d::Mahalanobis) = Ref(d)

logit(x) = @. log(x / (1 - x))
invlogit(x) = @. 1 / (1 + exp(-x))

"""
    optimal_bandwidth(X, y; distance, K, binary) -> h

Pick a kernel bandwidth by minimizing held-out SSE on an 80% subsample. Returns
the bandwidth in the same units as the kernel argument.
"""
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

"""
    npregress(X, y, Xstar = X; h, distance, K, binary)

Locally-weighted polynomial regression with kernel `K` and `distance`. For
`binary=true`, runs an IRLS loop on a logit link.

Returns a matrix whose rows are eval points and whose pairs of columns are
`[β̂_k, var(β̂_k)]` for k ∈ {intercept, slopes...}. For binary fits the columns
collapse to `[μ̂, var(μ̂)]`.
"""
function npregress(
    X,
    y,
    Xstar = X;
    h = nothing,
    distance = mahalanobis(X),
    K = pdf$Normal(),
    binary = false,
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
