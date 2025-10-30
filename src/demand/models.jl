abstract type AbstractCoefType end
abstract type AbstractDemandModel end

struct RandomCoefType <: AbstractCoefType
    draws::Vector{Float32} # S-vector
    grid::SparseArray{Float32, 3} # S x T × (J * F * K + 1), last slice is outside good
end
function RandomCoefType(S::Int64, T, J::Int64)
    return RandomCoefType(rand(Normal(0, 1), S), SparseArray{Float32}(undef, (S, T, J + 1)))
end

struct FixedCoefType <: AbstractCoefType
    draws::Vector{Float32} # 1-vector
    grid::SparseArray{Float32, 3} # 1 x T × (J * F * K + 1), last column is outside good
end
function FixedCoefType(::Int64, T::Int64, J::Int64)
    return FixedCoefType(zeros(Float32, 1), SparseArray{Float32}(undef, (1, T, J + 1)))
end
FixedCoefType(J::Int64) = FixedCoefType(1, J)

struct DemandModel{T <: AbstractCoefType} <: AbstractDemandModel
    u::T
    distances::NamedMatrix{Float32} # zips in rows, dealers in columns
    product_info::SparseArray{ProductInfo, 4} #  t, j, k, f
    product_lookup::ProductLookup
    agent_data::Dict{String, AgentInfo} # demographics for each zip code, sorted by zip
    counts::SparseArray{Int16, 5} # z, t, j, k, f
    counts_lookup::CountsLookup
end
function DemandModel{T}(;
    distances,
    product_info,
    product_lookup,
    agent_data,
    counts,
    counts_lookup,
) where {T <: AbstractCoefType}
    u = T(
        250, # number of simulation draws
        size(product_info, 1), # number of (quarter, dealer, model, trim) combinations
        prod(size(product_info)[2:4]),
    )
    return DemandModel{T}(
        u,
        distances,
        product_info,
        product_lookup,
        agent_data,
        counts,
        counts_lookup,
    )
end

# number of (quarter, dealer, model) combinations
n_delta(model::DemandModel) = length(keys(model.product_lookup.tjf))
n_trim_params(model::DemandModel) = 2length(unique(model.product_lookup.make))
n_demographics(_) = length(DEMOGRAPHIC_ATTRIBUTES) # median income, urban, pct_black, pct_hispanic
n_demo_product_params(x) = n_demographics(x) * n_product_attributes(x) # median income, urban, pct_black, pct_hispanic
n_product_attributes(_) = length(PRODUCT_ATTRIBUTES) # suv, truck, electric
n_demo_price_params(x) = n_demographics(x) # median income, urban
n_random_coefs(::DemandModel{RandomCoefType}) = 1 # price
n_random_coefs(::DemandModel{FixedCoefType}) = 0
n_dist_params(_) = 2 # distance disutility
n_mean_product_params(x) = n_product_attributes(x)
n_mean_price_params(_) = 1 # price
n_rho(_) = 1

const ORDER = [
    n_delta,
    n_trim_params,
    n_demo_product_params,
    n_demo_price_params,
    n_random_coefs,
    n_dist_params,
    n_mean_product_params,
    n_mean_price_params,
    n_rho,
]

total_n_params(model::DemandModel) = sum(f(model) for f in ORDER)
function glorot_initialization(model)
    return Float32.(
        rand(
            Uniform(-6 / total_n_params(model), 6 / total_n_params(model)),
            total_n_params(model),
        )
    )
end

macro pull(name, params, model)
    fname = Symbol("n_$(string(name))")
    quote
        i = findfirst(==($fname), ORDER)
        n_before = mapreduce(+, first(ORDER, i - 1), init = 0) do f
            f($(esc(model)))
        end
        @view $(esc(params))[n_before .+ (1:$fname($(esc(model))))]
    end
end

function δ(params::AbstractVector, model::AbstractDemandModel)
    guesses = @pull delta params model # only j * t * f (lacking trims)

    deltas = similar(model.product_info, Float32)

    for (idxs, delta) in zip(values(model.product_lookup.tjf), guesses)
        deltas[idxs] .= delta
    end

    # adjust by trim parameters (trim ≥ 0)
    θs = @pull trim_params params model

    nth(x, n) = last(first(x, n))
    for (θ, idxs) in zip(partition(θs, 2), values(model.product_lookup.make))
        iks = nth.(Tuple.(idxs), 3) # get trim ranks for these indices
        ks = nth.(Ref(keys(model.product_lookup.k)), iks)
        deltas[idxs] .*= @. 1 + quantile(Gamma(abs.(θ)...), ks)
    end

    return reshape(deltas, 1, size(deltas)...) # 1 × T × J × K × F
end

@memoize function x_matrix(model)
    out = SparseArray{Float32}(undef, (size(model.product_info)..., n_product_attributes(model) + 1)) # splat product info into last dimension
    for idx in nonzero_keys(model.product_info)
        info = model.product_info[idx]
        out[Tuple(idx)..., :] .= [info[PRODUCT_ATTRIBUTES]..., info.p]
    end
    return reshape(out, 1, size(out)...)
end

# demographic/product attribute interaction, demographic/price and random coeff interaction, distance disutility
function μ(params, model::DemandModel{FixedCoefType}, zip_code)
    Πᵦ = @pull demo_product_params params model # (n_demo x n_product)
    Πᵦ = reshape(Πᵦ, n_demographics(model), n_product_attributes(model)) # (n_demo × n_product)
    Πₚ = @pull demo_price_params params model # (n_demo)
    η = @pull dist_params params model # (2)
    Π = hcat(Πᵦ, Πₚ) # n_demo x (n_product + 1)

    x_j = x_matrix(model) # T × J × K × F × (n_product + price)
    D_i = collect(model.agent_data[zip_code][DEMOGRAPHIC_ATTRIBUTES]) # n_demo-vector

    dist_f = model.distances[zip_code, :] # 1 x F

    # start with demographic utility over price and product attributes
    out = tensorcontract(x_j, (1, 2, 3, 4, 5, -1), SparseArray(Π'D_i), (-1,))

    # add distance disutility
    for (uid, dist) in zip(names(dist_f, 1), dist_f)
        idxs = model.product_lookup.f[uid]
        for idx in idxs
            out[1, Tuple(idx)...] -= exp.(η[1] .+ η[2] * dist)
        end
    end

    # do not add random coefficients :)

    return out
end

# μ_outsize(model::DemandModel{FixedCoefType}) = (1, size(model.product_info)...)
# μ_outsize(model::DemandModel{RandomCoefType}) = (length(model.u.draws), size(model.product_info)...)

# randomutility(_, ::DemandModel{FixedCoefType}) = 0
# function randomutility(params, model::DemandModel{RandomCoefType})
#     x_j = x_matrix(model) # 1 × T × J × K × F × (n_product + price)
#     σ = @pull random_coefs params model
#     return model.u.draws * abs(σ) .* x_j[:, :, :, :, :, 2]
# end

# loglikelihood for all models, trims, quarters, dealers
function loglikelihood(
    model::AbstractDemandModel, params::AbstractVector, stochastic::Bool = true
)
    zips = if stochastic
        sample(collect(keys(model.agent_data)), length(model.agent_data) ÷ 10, replace = false) # sample 10% of zip codes for likelihood calculation
    else
        collect(keys(model.agent_data))
    end
    δ̂ = δ(params, model) # include mean utilities and adjustment for trim 1 × T × (J * F * K)
    return Folds.mapreduce(+, zips, ThreadedEx(simd=true)) do z
        grid = δ̂ + μ(params, model, z) # (S x T × J × F × K)
        CCPs!(model, grid) # choice probabilities (S x J * T * F)
        # TODO: take mean across first dimension with tensorcontract
        grid_keys = collect(nonzero_keys(grid))
        grid[grid_keys] = log.(grid[grid_keys])
        iz = searchsortedfirst(collect(keys(model.counts_lookup.z)), z)
        counts = reshape(model.counts[iz, :, :, :, :], :)
        dot(counts, reshape(grid, :)) # log likelihood of all models, trims, quarters, firms for zip z
    end
end

function CCPs!(model, grid)
    for ts in values(model.product_lookup.t)
        for i in 1:size(grid, 1)
            idxs = map(t -> CartesianIndex(i, t...), Tuple.(ts))
            u = maximum(grid[idxs])
            outside_option = exp(-u)
            grid[idxs] .= exp.(grid[idxs] .- u)
            grid[idxs] ./= sum(grid[idxs]) + outside_option
        end
    end
    return nothing
end
