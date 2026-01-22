function δ(params::AbstractVector, model::DemandModel)::SparseArray{Float32, 4}
    guesses::Vector{Float32} = @pull delta params model # only t * j * f (lacking trims)

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

    return deltas # T × J × K × F
end

# demographic/product attribute interaction, demographic/price and random coeff interaction, distance disutility
function μ(params, model::AbstractDemandModel, zip_code)::SparseArray{Float32, 5}
    Πᵦ::Matrix{Float32} = reshape(
        @pull(demo_product_params, params, model),
        n_demographics(model),
        n_product_attributes(model),
    ) # (n_demo × n_product)
    Πₚ = @pull demo_price_params params model # (n_demo)
    η = @pull dist_params params model # (2)
    Π = hcat(Πᵦ, Πₚ) # n_demo x (n_product + 1)

    x_j = x_matrix(model) # T × J × K × F × (n_product + price)
    x_j = reshape(x_j, 1, size(x_j)...) # 1 × T × J × K × F × (n_product + price)
    D_i::Vector{Float32} = collect(model.agent_data[zip_code][DEMOGRAPHIC_ATTRIBUTES]) # n_demo-vector

    dist_f = model.distances[zip_code, :] # 1 x F

    # start with demographic utility over price and product attributes
    out::SparseArray{Float32, 5} = tensorcontract(
        x_j, (1, 2, 3, 4, 5, -1), SparseArray(Π'D_i), (-1,)
    )

    # add distance disutility
    @inbounds for (uid, dist) in zip(names(dist_f, 1), dist_f)
        # regular getindex dispatches to a method that can sometimes resize. Not thread safe!
        idxs = model.product_lookup.f[uid]
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
            out_keys = Base.KeySet{CartesianIndex{5}, Dict{CartesianIndex{5}, Float32}}(
                nonzero_keys(out)
            )
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

# loglikelihood for all models, trims, quarters, dealers
function loglikelihood(
    model::AbstractDemandModel,
    params::AbstractVector,
    z::String,
)::Float32
    mean_vec = SparseArray{Float32, 1}(ones(Float32, size(model.u.draws)))
    mean_vec /= Float32(size(model.u.draws, 1))

    grid = CCPs(model, params, z) # S x T × J × K × F

    # Take mean across first dimension
    grid = tensorcontract(mean_vec, (-1,), grid, (-1, 2, 3, 4, 5))

    # log mean choice probabilities
    grid_keys::Vector{CartesianIndex{4}} = collect(nonzero_keys(grid))
    @inbounds grid[grid_keys] = log.(grid[grid_keys])
    grid_reshape::SparseArray{Float32, 1} = reshape(grid, :)

    # pull counts for zip
    iz = searchsortedfirst(collect(keys(model.counts_lookup.z)), z)
    counts::SparseArray{Float32, 1} = reshape(model.counts[iz, :, :, :, :], :)

    # log likelihood of all models, trims, quarters, firms for zip
    return dot(counts, grid_reshape)
end

function CCPs(model, params, z)
    # mean utilities and trim adjustments (T × J × K × F, dropping nothing)
    δ̂ = δ(params, model)
    δ̂ = reshape(δ̂, 1, size(δ̂)...)
    # utility up to idiosyncratic error (S x T × J × K × F)
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

    # choice probabilities (S x T × J × K × F)
    CCPs!(model, grid)

    return grid
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

function ξ̂(model::AbstractDemandModel, params::AbstractVector)::SparseArray{Float32, 4}
    # mean utilities and trim adjustments (T × J × F, dropping K)
    δ̂ = δ(params, model)
    residualize!(δ̂, model.product_lookup) # in-place residualization

    x_j = x_residualized(model) # T × J × F × (n_product + price), dropping K

    βₓ = @pull mean_product_params params model # (n_product)
    βₚ = only(@pull mean_price_params params model) # (1)

    # not decomposing for now
    # ρ = only(@pull rho params model)
    # γ = @pull group_fes params model # (n_group_fes)

    # Ω, i_Ω = Ω_matrix(model.ownership, model.adjacency)
    # @inbounds @simd for idx in i_Ω
    #     t, j, f, g = idx.I
    #     Ω[t, j, f, g] = model.ownership[t, f, g] + ρ * model.adjacency[t, j, f, g]
    # end

    linear_params = tensorcontract(x_j, (1, 2, 3, 4, -1), SparseArray(vcat(βₓ, βₚ)), (-1,))

    # @inbounds for i in axes(Ω, 4)
    #     @info "Iteration $i of $(size(Ω, 4)) residualizing Ω matrix"
    #     linear_params += residualize!(Ω[:, :, :, i], model.product_lookup) * γ[i]
    # end

    return δ̂ - linear_params
end

function moments(model::AbstractDemandModel, params::AbstractVector)::AbstractVector
    ξ = ξ̂(model, params)
    Z = z_residualized(model)
    return mapreduce(vcat, axes(Z, 5)) do i_demo
        mapreduce(+, nonzero_keys(ξ)) do idx
            @inbounds ξ[idx] * Z[Tuple(idx)..., i_demo]
        end
    end
end

function Π̂(model::AbstractDemandModel, params::AbstractVector, W = I)::Float32
    m = moments(model, params)
    return 0.5f0 * only(m' * W * m)
end

function dΠ̂(model::AbstractDemandModel, params::AbstractVector, W = I)::AbstractVector
    m = moments(model, params)
    ∇m = reduce(hcat, (zeros(Float32, size(params)) for _ in 1:length(m)))

    i_δ = @indexof delta params model
    δs = δ(params, model)
    i_θ = @indexof trim_params params model
    θs = @pull trim_params params model

    Z = z_matrix(model)

    nth = last ∘ first

    # grad for delta
    @inbounds for (i, iter) in enumerate(pairs(model.product_lookup.tjf))
        (tjf, idxs) = iter
        make = tjf[2][1]
        imake = searchsortedfirst(collect(keys(model.product_lookup.make)), make)
        θ = nth(partition(θs, 2), imake)
        iks = nth.(Tuple.(idxs), 3) # get trim ranks for these indices
        ks = nth.(Ref(keys(model.product_lookup.k)), iks)
        trim_factor = 1 .+ quantile.(Gamma(abs.(θ)...), ks)
        @inbounds for col in axes(∇m, 2)
            row = i_δ[i]
            ∇m[row, col] = sum(t * Z[Tuple(idx)..., col] for (t, idx) in zip(trim_factor, idxs))
        end
    end

    # grad for beta
    X = x_matrix(model)
    i_β = @indexof mean_product_params params model
    i_β_price = only(@indexof mean_price_params params model)
    i_β = vcat(i_β, i_β_price)
    @inbounds for z_var in axes(Z, 5)
        @inbounds for x_var in axes(X, 5)
            row = i_β[x_var]
            col = z_var
            ∇m[row, col] = sum(
                @inbounds X[Tuple(idx)..., x_var] * Z[Tuple(idx)..., z_var]
                for idxs in nonzero_keys(model.product_info)
            )
        end
    end

    cache = GradientCache(0.0f0, zeros(Float32, 2), Val(:forward))
    # grad for trim params
    @inbounds for (i, (θ, idxs)) in
                  enumerate(zip(partition(θs, 2), values(model.product_lookup.make)))
        iks = nth.(Tuple.(idxs), 3) # get trim ranks for these indices
        ks = nth.(Ref(keys(model.product_lookup.k)), iks)
        k_dict = DefaultOrderedDict{Float32, Vector{CartesianIndex{4}}}(Vector{CartesianIndex{4}})
        @inbounds for (idx, k) in zip(idxs, ks)
            push!(k_dict[k], idx)
        end
        @inbounds for (k, idxs_k) in pairs(k_dict)
            i_θᵢ = nth(partition(i_θ, 2), i)
            grad = finite_difference_gradient(θ -> quantile.(Gamma(abs.(θ)...), k), θ, cache)
            @inbounds for col in axes(Z, 5)
                ∇m[i_θᵢ, col] += sum(δs[idx] * grad * Z[Tuple(idx)..., col] for idx in idxs_k)
            end
        end
    end

    return 0.5f0 * ∇m * W * m
end

function dloglikelihood(
    model::AbstractDemandModel,
    params::AbstractVector,
    z::String,
)::AbstractVector
    mean_vec = SparseArray{Float32, 1}(ones(Float32, size(model.u.draws)))
    mean_vec /= Float32(size(model.u.draws, 1))

    grid = CCPs(model, params, z) # S x T × J × K × F

    # Take mean across first dimension
    grid = tensorcontract(mean_vec, (-1,), grid, (-1, 2, 3, 4, 5))

    # pull counts for zip
    iz = searchsortedfirst(collect(keys(model.counts_lookup.z)), z)
    counts = model.counts[iz, :, :, :, :]
    t_counts = SparseArray{Int64}(undef, size(counts, 1))
    @inbounds for i in nonzero_keys(counts)
        t_counts[first(i.I)] += counts[i]
    end

    out = zeros(Float32, length(params))

    i_δ = @indexof delta params model
    δs = δ(params, model)
    i_θ = @indexof trim_params params model
    θs = @pull trim_params params model

    # grad for delta
    @inbounds for (i, iter) in enumerate(pairs(model.product_lookup.tjf))
        (tjf, idxs) = iter
        make = tjf[2][1]
        imake = searchsortedfirst(collect(keys(model.product_lookup.make)), make)
        θ = nth(partition(θs, 2), imake)
        iks = nth.(Tuple.(idxs), 3) # get trim ranks for these indices
        ks = nth.(Ref(keys(model.product_lookup.k)), iks)
        trim_factor = 1 .+ quantile.(Gamma(abs.(θ)...), ks)
        t = idxs[1].I[1]
        out[i_δ[i]] = sum(trim_factor .* (counts[idxs] .- t_counts[t] * grid[idxs]))
    end

    # grad for trim params
    cache = GradientCache(0.0f0, zeros(Float32, 2), Val(:forward))
    @inbounds for (i, (θ, idxs)) in
                  enumerate(zip(partition(θs, 2), values(model.product_lookup.make)))
        iks = nth.(Tuple.(idxs), 3) # get trim ranks for these indices
        ks = nth.(Ref(keys(model.product_lookup.k)), iks)
        k_dict = DefaultOrderedDict{Float32, Vector{CartesianIndex{4}}}(Vector{CartesianIndex{4}})
        @inbounds for (idx, k) in zip(idxs, ks)
            push!(k_dict[k], idx)
        end
        @inbounds for (k, idxs_k) in pairs(k_dict)
            i_θᵢ = nth(partition(i_θ, 2), i)
            grad = finite_difference_gradient(θ -> quantile.(Gamma(abs.(θ)...), k), θ, cache)
            out[i_θᵢ] = sum(
                δs[idx] * grad * (counts[idx] - t_counts[first(idx.I)] * grid[idx])
                for idx in idxs_k
            )
        end
    end

    # grade for demo_produce and demo_price_params
    X = x_matrix(model)
    i_Πᵦ = reshape(
        @indexof(demo_product_params, params, model),
        n_demographics(model),
        n_product_attributes(model),
    ) # (n_demo × n_product)
    i_Πₚ = @indexof demo_price_params params model # (n_demo)
    i_Π = hcat(i_Πᵦ, i_Πₚ) # n_demo x (n_product + 1)
    D_i = collect(model.agent_data[z][DEMOGRAPHIC_ATTRIBUTES])
    @inbounds for i_X in axes(X, 5)
        for i_D in axes(D_i, 1)
            row = i_Π[i_D, i_X]
            out[row] = sum(
                D_i[i_D] * X[Tuple(idx)..., i_X] * (counts[idx] - t_counts[first(idx.I)] * grid[idx])
                for idx in nonzero_keys(model.product_info)
            )
        end
    end

    η = @pull dist_params params model
    i_η = @indexof dist_params params model
    dist_f = model.distances[z, :]
    @inbounds for (uid, dist) in zip(names(dist_f, 1), dist_f)
        idxs = model.product_lookup.f[uid]
        @inbounds for idx in idxs
            out[i_η[1]] -= exp(η[1] .+ η[2] * dist) *
                (counts[idx] - t_counts[first(idx.I)] * grid[idx])
            out[i_η[2]] -= dist * exp(η[1] .+ η[2] * dist) *
                (counts[idx] - t_counts[first(idx.I)] * grid[idx])
        end
    end
end
