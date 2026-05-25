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
    demo_scales::Vector{Float32}   # std dev of each demographic in DEMOGRAPHIC_ATTRIBUTES order
    price_scale::Float32           # std dev of price across all (t,j,f) observations
    dist_scale::Float32            # std dev of distance across all (zip,dealer) pairs in miles
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
    demo_scales,
    price_scale,
    dist_scale,
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
        demo_scales,
        price_scale,
        dist_scale,
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
    # n_mean_price_params,
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

function _fe_linear_map(t_refs, j_refs, f_refs, n_t, n_j, n_f)
    N = length(t_refs)
    M = Int(n_t + n_j + n_f)
    fwd! = (y, x) -> begin
        fill!(y, zero(Float32))
        @fastmath @inbounds for i in 1:N
            y[i] = x[t_refs[i]] + x[n_t + j_refs[i]] + x[n_t + n_j + f_refs[i]]
        end
        return y
    end
    adj! = (y, x) -> begin
        fill!(y, zero(Float32))
        @fastmath @inbounds for i in 1:N
            xi = x[i]
            y[t_refs[i]] += xi
            y[n_t + j_refs[i]] += xi
            y[n_t + n_j + f_refs[i]] += xi
        end
        return y
    end
    return LinearMap{Float32}(fwd!, adj!, N, M; ismutating = true)
end

@memoize function _residualize_plan(lookup)
    t_lookup, j_lookup, f_lookup = collect_lookups(lookup)
    obs_keys = reduce(vcat, t_lookup)
    N = length(obs_keys)
    key_to_obs = Dict(k => Int32(i) for (i, k) in enumerate(obs_keys))

    t_refs = Vector{Int32}(undef, N)
    j_refs = Vector{Int32}(undef, N)
    f_refs = Vector{Int32}(undef, N)
    for (gi, group) in enumerate(t_lookup)
        for k in group
            t_refs[key_to_obs[k]] = Int32(gi)
        end
    end
    for (gi, group) in enumerate(j_lookup)
        for k in group
            j_refs[key_to_obs[k]] = Int32(gi)
        end
    end
    for (gi, group) in enumerate(f_lookup)
        for k in group
            f_refs[key_to_obs[k]] = Int32(gi)
        end
    end

    A = _fe_linear_map(t_refs, j_refs, f_refs, Int32(length(t_lookup)), Int32(length(j_lookup)), Int32(length(f_lookup)))
    ws = LsmrWorkspace(N, size(A, 2), Vector{Float32})
    return obs_keys, A, ws
end

function residualize!(X, lookup)
    obs_keys, A, ws = _residualize_plan(lookup)
    N = length(obs_keys)
    b = Vector{Float32}(undef, N)
    @inbounds for (i, k) in enumerate(obs_keys)
        b[i] = X[k]
    end
    lsmr!(ws, A, b; atol = 1f-4, btol = 1f-4, itmax = 500)
    ws.stats.solved || @warn "residualize! did not converge in $(ws.stats.niter) iterations"
    @info "residualize! solved in $(ws.stats.niter) LSMR iterations"
    fitted = ws.Av  # reuse workspace buffer (length m = N)
    mul!(fitted, A, ws.x)
    @inbounds for (i, k) in enumerate(obs_keys)
        X[k] = b[i] - fitted[i]
    end
    return X
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
    out = zeros(Float32, total_n_params(model))

    # δ: target P(outside) ≈ 0.99 (most people don't buy cars).
    # With J products per market and P(outside) = 0.99:
    #   J × exp(V_j) = 0.01  →  V_j = log(0.01/J) ≈ -12 for J≈1500.
    # V_j = δ + μ; μ ≈ -exp(η₁) ≈ -1 at zero distance with η₁=0 below.
    # So δ₀ ≈ -12 + 1 = -11.  Use -10 as a round number.
    out[@indexof delta out model] .= -10.0f0

    # Distance disutility: exp(η₁ + η₂ × dist') where dist' = dist_miles / σ_dist.
    # η₁=-4 → baseline cost exp(-4)≈0.018 at d=0 (unchanged by standardization).
    # η₂: old 0.01/mile × σ_dist ≈ 1.0 per std-dev (σ_dist ≈ 100 miles).
    # After standardization |∂ℓ/∂η₂| / |∂ℓ/∂η₁| ≈ mean(dist') = O(1).
    out[@indexof dist_params out model] .= Float32[-3.0f0, 2.0f0]

    # β in ξ̂ = δ_res - β×p_res where p_res is residualized price in DOLLARS.
    # With residualized price std ≈ $5000, β=-1e-4 → ≈0.5 utility units.
    # Previous β=-0.1 → 500 utility units per $5K variation, far too large.
    # out[@indexof mean_price_params out model] .= -1.0f-3

    # demo_product_params (Πᵦ, 4×4): demographics vary fastest, products slowest
    # (ORDER = [coupe, suv, pickup, sedan] × [income, urban, black, hispanic]).
    # Observed gradients at Πᵦ=0 are all negative, so we start positive.
    # Pickup has the largest effect, suv medium, sedan small, coupe smallest.
    out[@indexof demo_product_params out model] .=
        repeat(Float32[1.1f0, 1.5f0, 2.0f0, 1.25f0], inner = n_demographics(model))

    # demo_price_params (Πₚ, length 4): demo × price interactions.
    # After standardization, Πₚ[d] × D'_z[d] × p' has natural scale O(1) for all d.
    # Uniform starting value -0.5 for all demographics.
    out[@indexof demo_price_params out model] .= -1.5f0

    return out
end
