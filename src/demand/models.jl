abstract type AbstractCoefType end
abstract type AbstractDemandModel end

struct RandomCoefType <: AbstractCoefType
    draws::Vector{Float32} # S-vector
end
RandomCoefType(S::Int64, T, J::Int64) = RandomCoefType(rand(Normal(0, 1), S))

struct FixedCoefType <: AbstractCoefType
    draws::Vector{Float32} # 1-vector
end
FixedCoefType(::Int64, T::Int64, J::Int64) = FixedCoefType(zeros(Float32, 1))
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
        100, # number of simulation draws
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

@memoize function x_matrix(model)
    out = SparseArray{Float32}(
        undef, (size(model.product_info)..., n_product_attributes(model) + 1)
    ) # splat product info into last dimension
    for idx in nonzero_keys(model.product_info)
        info = model.product_info[idx]
        out[Tuple(idx)..., :] .= [info[PRODUCT_ATTRIBUTES]..., info.p]
    end
    return reshape(out, 1, size(out)...)
end

function glorot_initialization(model)
    return Float32.(
        rand(
            Uniform(-6 / total_n_params(model), 6 / total_n_params(model)),
            total_n_params(model),
        )
    )
end
