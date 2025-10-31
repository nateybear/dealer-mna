function δ(params::AbstractVector, model::DemandModel)::SparseArray{Float32, 5}
    guesses::Vector{Float32} = @pull delta params model # only j * t * f (lacking trims)

    deltas = similar(model.product_info, Float32)

    @inbounds for (idxs, delta) in zip(values(model.product_lookup.tjf), guesses)
        deltas[idxs] .= delta
    end

    # adjust by trim parameters (trim ≥ 0)
    θs::Vector{Float32} = @pull trim_params params model

    nth(x::Union{NTuple{N, T}, AbstractSet{T}}, n::Int64) where {N, T} = last(first(x, n))::T
    @inbounds for (θ, idxs) in zip(partition(θs, 2), values(model.product_lookup.make))
        iks = nth.(Tuple.(idxs), 3) # get trim ranks for these indices
        @inline ks = nth.(Ref(keys(model.product_lookup.k)), iks)
        deltas[idxs] .*= 1 .+ quantile.(Gamma(abs.(θ)...), ks)
    end

    return reshape(deltas, 1, size(deltas)...) # 1 × T × J × K × F
end

# demographic/product attribute interaction, demographic/price and random coeff interaction, distance disutility
function μ(params, model::AbstractDemandModel, zip_code)::SparseArray{Float32, 5}
    Πᵦ::Matrix{Float32} = reshape(@pull(demo_product_params, params, model), n_demographics(model), n_product_attributes(model)) # (n_demo × n_product)
    Πₚ = @pull demo_price_params params model # (n_demo)
    η = @pull dist_params params model # (2)
    Π = hcat(Πᵦ, Πₚ) # n_demo x (n_product + 1)

    x_j = x_matrix(model) # T × J × K × F × (n_product + price)
    D_i = Vector{Float32}(model.agent_data[zip_code][DEMOGRAPHIC_ATTRIBUTES]) # n_demo-vector

    dist_f = model.distances[zip_code, :] # 1 x F

    # start with demographic utility over price and product attributes
    out::SparseArray{Float32, 5} = tensorcontract(x_j, (1, 2, 3, 4, 5, -1), SparseArray(Π'D_i), (-1,))

    # add distance disutility
    @inbounds for (uid, dist) in zip(names(dist_f, 1), dist_f)
        idxs = lock(MUTEX) do
            model.product_lookup.f[uid] # sometimes can resize? lock for memory safety
        end
        for idx in idxs
            out[1, Tuple(idx)...] -= exp.(η[1] .+ η[2] * dist)
        end
    end

    # add random utility (or zero for fixed coef model)
    # S × T × J × K × F
    ru = randomutility(params, model)
    S = size(ru, 1)

    if S > 1
        # Expand out to S dimensions and add random utility only to existing sparse entries
        expanded_out = similar(ru)

        # Copy existing entries and add random utility
        @inbounds for s in 1:S
            out_keys = Base.KeySet{CartesianIndex{5}, Dict{CartesianIndex{5}, Float32}}(nonzero_keys(out))
            for idx in out_keys
                (_, i...) = Tuple(idx)
                expanded_out[s, i...] = Float32(out[idx])
            end
        end

        expanded_out += ru

        return expanded_out
    else
        return out
    end
end

function randomutility(_, ::DemandModel{FixedCoefType})::SparseArray{Float32, 5}
    return reshape(SparseArray([0.0f0]), ones(Int8, 5)...)
end
function randomutility(params, model::DemandModel{RandomCoefType})::SparseArray{Float32, 5}
    σ = @pull random_coefs params model # (1) for price
    σ = abs(only(σ))
    p = x_matrix(model)[:, :, :, :, :, end] # extract price from x_matrix
    S = length(model.u.draws)
    out = SparseArray{Float32}(undef, (S, size(p)[2:end]...))
    @inbounds for s in 1:S
        for idx in nonzero_keys(p)
            (_, i...) = Tuple(idx)
            out[s, i...] = p[idx] * σ * model.u.draws[s]
        end
    end
    return out
end

function with_blas_single_threading(f)
    old = BLAS.get_num_threads()
    BLAS.set_num_threads(1)
    try
        f()
    finally
        BLAS.set_num_threads(old)
    end
end

# loglikelihood for all models, trims, quarters, dealers
function loglikelihood(
    model::AbstractDemandModel, 
    params::AbstractVector, 
    stochastic::Bool = true
)::Float32
    zips::Vector{String} = if stochastic
        sample(collect(keys(model.agent_data)), length(model.agent_data) ÷ 10, replace = false) # sample 10% of zip codes for likelihood calculation
    else
        collect(keys(model.agent_data))
    end

    # mean utilities and trim adjustments (1 × T × J × F × K)
    δ̂ = δ(params, model)

    mean_vec = SparseArray{Float32, 1}(ones(Float32, size(model.u.draws)))
    mean_vec /= Float32(size(model.u.draws, 1))

    with_blas_single_threading() do
        Folds.mapreduce(+, zips, WorkStealingEx(simd = true)) do z
            # utility up to idiosyncratic error (S x T × J × F × K)
            μ̂ = μ(params, model, z)

            grid = if size(μ̂, 1) > 1
                out = similar(μ̂)
                @inbounds for s in 1:size(μ̂, 1)
                    for idx in nonzero_keys(δ̂)
                        (_, i...) = Tuple(idx)
                        out[s, i...] = δ̂[1, i...]
                    end
                end
                out + μ̂
            else
                δ̂ + μ̂
            end

            # choice probabilities (S x T × J × F × K)
            CCPs!(model, grid)

            # Take mean across first dimension
            grid::SparseArray{Float32, 4} = tensorcontract(mean_vec, (-1,), grid, (-1, 2, 3, 4, 5))

            # log mean choice probabilities
            grid_keys = Vector{CartesianIndex{4}}(nonzero_keys(grid))
            grid[grid_keys] = log.(grid[grid_keys])
            grid_reshape::SparseVector{Float32} = reshape(grid, :)

            # pull counts for zip z
            iz = searchsortedfirst(collect(keys(model.counts_lookup.z)), z)
            counts::SparseVector{Float32} = reshape(model.counts[iz, :, :, :, :], :)

            # log likelihood of all models, trims, quarters, firms for zip z
            Float32(dot(counts, grid_reshape))
        end
    end
end

function CCPs!(model, grid)
    @inbounds for ts in values(model.product_lookup.t)
        for i in 1:size(grid, 1)
            idxs = map(t -> CartesianIndex(i, t...), Tuple.(ts))
            u = maximum(grid[idxs])
            outside_option = exp(-u)
            grid[idxs] = exp.(grid[idxs] .- u)
            grid[idxs] /= sum(grid[idxs]) + outside_option
        end
    end
    return nothing
end
