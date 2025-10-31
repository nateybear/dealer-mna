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
        deltas[idxs] .*= 1 .+ quantile.(Gamma(abs.(θ)...), ks)
    end

    return reshape(deltas, 1, size(deltas)...) # 1 × T × J × K × F
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
        idxs = lock(MUTEX) do
            model.product_lookup.f[uid] # sometimes can resize? lock for memory safety
        end
        for idx in idxs
            out[1, Tuple(idx)...] -= exp.(η[1] .+ η[2] * dist)
        end
    end

    # do not add random coefficients :)

    return out
end

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
    blas_threads = BLAS.get_num_threads()
    BLAS.set_num_threads(1)
    out = Folds.mapreduce(+, zips, WorkStealingEx(simd = true)) do z
        grid = δ̂ + μ(params, model, z) # (S x T × J × F × K)
        CCPs!(model, grid) # choice probabilities (S x J * T * F)
        # TODO: take mean across first dimension with tensorcontract
        grid_keys = collect(nonzero_keys(grid))
        grid[grid_keys] = log.(grid[grid_keys])
        iz = searchsortedfirst(collect(keys(model.counts_lookup.z)), z)
        counts = reshape(model.counts[iz, :, :, :, :], :)
        dot(counts, reshape(grid, :)) # log likelihood of all models, trims, quarters, firms for zip z
    end
    BLAS.set_num_threads(blas_threads)
    return out
end

function CCPs!(model, grid)
    for ts in values(model.product_lookup.t)
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
