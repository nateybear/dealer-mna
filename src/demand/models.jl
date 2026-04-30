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
    product_info::SparseArray{ProductInfo, 3} #  t, j, f
    product_lookup::ProductLookup
    agent_data::Dict{String, AgentInfo} # demographics for each zip code, sorted by zip
    counts::SparseArray{Int16, 4} # z, t, j, f
    counts_lookup::CountsLookup
    ownership::SparseArray{Bool, 3} # t, f, g
    ownership_lookup::OwnershipLookup
end
function DemandModel{T}(;
    distances,
    product_info,
    product_lookup,
    agent_data,
    counts,
    counts_lookup,
    ownership,
    ownership_lookup,
) where {T <: AbstractCoefType}
    u = T(
        100, # number of simulation draws
        size(product_info, 1), # number of (quarter, model-trim, dealer) combinations
        prod(size(product_info)[2:3]),
    )
    return DemandModel{T}(
        u,
        distances,
        product_info,
        product_lookup,
        agent_data,
        counts,
        counts_lookup,
        ownership,
        ownership_lookup,
    )
end

# number of (quarter, model-trim, dealer) combinations
n_delta(model::DemandModel) = length(keys(model.product_lookup.tjf))
n_demographics(_) = length(DEMOGRAPHIC_ATTRIBUTES) # median income, urban, pct_black, pct_hispanic
n_demo_product_params(x) = n_demographics(x) * n_product_attributes(x) # median income, urban, pct_black, pct_hispanic
n_product_attributes(_) = length(PRODUCT_ATTRIBUTES) # suv, truck, electric
n_demo_price_params(x) = n_demographics(x) # median income, urban
n_random_coefs(::DemandModel{RandomCoefType}) = 1 # price
n_random_coefs(::DemandModel{FixedCoefType}) = 0
n_dist_params(_) = 2 # distance disutility
n_mean_price_params(_) = 1 # price

const ORDER = [
    n_delta,
    n_demo_product_params,
    n_demo_price_params,
    n_random_coefs,
    n_dist_params,
    n_mean_price_params,
]

total_n_params(model::DemandModel) = sum(f(model) for f in ORDER)

macro indexof(name, params, model)
    fname = Symbol("n_$(string(name))")
    quote
        i = findfirst(==($fname), ORDER)
        n_before = mapreduce(+, first(ORDER, i - 1), init = 0) do f
            f($(esc(model)))
        end
        n_current = $fname($(esc(model)))
        n_before .+ (1:n_current)
    end
end

macro pull(name, params, model)
    quote
        i = @indexof($name, $(esc(params)), $(esc(model)))
        @view $(esc(params))[i]
    end
end

@memoize function x_matrix(model)
    out = SparseArray{Float32}(
        undef, (size(model.product_info)..., n_product_attributes(model) + 1)
    ) # splat product info into last dimension
    for idx in nonzero_keys(model.product_info)
        info = model.product_info[idx]
        out[Tuple(idx)..., :] .= [model.product_info[idx][PRODUCT_ATTRIBUTES]..., info.p]
    end
    return out
end

@memoize x_matrix_mu(model) =
    let X = x_matrix(model)
        reshape(X, 1, size(X)...)
    end

@memoize p_matrix(model) = x_matrix(model)[:, :, :, end:end]

@memoize function z_matrix(model)
    out = SparseArray{Float32}(
        undef, (size(model.product_info)..., 2)
    ) # splat product info into last dimension
    for idx in nonzero_keys(model.product_info)
        info = model.product_info[idx]
        out[Tuple(idx)..., :] .= [info.p_hausman, 1.0f0]
    end
    return out
end

@memoize function collect_lookups(lookup)
    t_lookup = collect.(values(lookup.t))
    j_lookup = collect.(values(lookup.j))
    f_lookup = collect.(values(lookup.f))
    return t_lookup, j_lookup, f_lookup
end

function residualize!(X, lookup)
    i_iter = 0
    t_lookup, j_lookup, f_lookup = collect_lookups(lookup)
    X_old = similar(X)
    while true
        i_iter += 1
        copy!(X_old, X)
        for t in t_lookup
            @inbounds X[t] .-= @fastmath mean(X[t])
        end
        for j in j_lookup
            @inbounds X[j] .-= @fastmath mean(X[j])
        end
        for f in f_lookup
            @inbounds X[f] .-= @fastmath mean(X[f])
        end
        rdiff = @fastmath norm(X - X_old) / norm(X_old)
        if rdiff < 1e-4
            @info "Terminating after $i_iter iterations"
            return X
        end
        i_iter % 10 == 0 && @info "Residualizing hit $i_iter iterations"
    end
end

@memoize function x_residualized(model)
    x_j = x_matrix(model)[:, :, :, end:end]
    @inbounds for i_demo in axes(x_j, 4)
        x_j[:, :, :, i_demo] = residualize!(x_j[:, :, :, i_demo], model.product_lookup)
    end
    return x_j
end

@memoize function z_residualized(model)
    z_j = copy(z_matrix(model))
    @inbounds for i_demo in axes(z_j, 4)[1:(end - 1)]
        z_j[:, :, :, i_demo] = residualize!(z_j[:, :, :, i_demo], model.product_lookup)
    end
    return z_j
end

# @memoize function Ω_matrix(ownership, adjacency)
#     Ω = similar(adjacency)
#     tj_combos = unique(idx.I[1:2] for idx in nonzero_keys(adjacency))
#     for (t, j) in tj_combos
#         @inbounds Ω[t, j, :, :] .= ownership[t, :, :] + adjacency[t, j, :, :]
#     end
#     return Ω, collect(nonzero_keys(Ω))
# end

function initial_parameters(model)
    out =
        Float32.(
            rand(
                Uniform(-2, 2),
                total_n_params(model),
            )
        )
    out[@indexof mean_price_params out model] .= -1.0f0
    out[@indexof dist_params out model] ./= 1.0f4
    all_price_params = @indexof demo_price_params out model # going to leave 1/income unscaled
    out[all_price_params[2:end]] ./= 1.0f3
    return out
end
