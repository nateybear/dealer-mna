@memoize δ_prealloc(model) = similar(model.product_info, Float32)

# Pre-warm the preallocated dict so all keys exist with non-zero sentinels, then record
# the exact hash-table slot for each (sparse_idx → guess_idx) pair.  On every subsequent
# call we scatter directly into deltas.data.vals[slots], bypassing the Dict API and its
# zero-deletion branch entirely — safe as long as we never add or remove keys afterwards.
@memoize function δ_fill_plan(model)
    deltas = δ_prealloc(model)
    for idxs in values(model.product_lookup.tjf)
        for idx in idxs
            deltas.data[idx] = one(Float32)  # non-zero sentinel keeps the slot alive
        end
    end
    n = sum(length(idxs) for idxs in values(model.product_lookup.tjf))
    slots = Vector{Int}(undef, n)
    gidxs = Vector{Int32}(undef, n)
    k = 1
    for (i, idxs) in enumerate(values(model.product_lookup.tjf))
        for idx in idxs
            slots[k] = Base.ht_keyindex(deltas.data, idx)
            gidxs[k] = Int32(i)
            k += 1
        end
    end
    return slots, gidxs
end

function δ(params::AbstractVector, model::DemandModel)::SparseArray{Float32, 3}
    guesses::Vector{Float32} = @pull delta params model # one delta per (t, j_trim, f)
    deltas = δ_prealloc(model)
    slots, gidxs = δ_fill_plan(model)
    dvals = deltas.data.vals
    @inbounds @simd for k in eachindex(slots)
        dvals[slots[k]] = guesses[gidxs[k]]
    end
    return deltas # T × J(trim-included) × F
end

# the interface is weird, make it easier
function contract!(into, A, IA, B, IB)
    IC = symdiff(IA, IB)
    pA, pB, pAB = contract_indices(IA, IB, IC)
    @no_escape begin
        tensorcontract!(
            into, A, pA, false, B, pB, false, pAB, 1, 0, DefaultBackend(),
            Bumper.default_buffer(),
        )
    end
    return nothing
end

@memoize μ_prealloc(s) = SparseArray{Float32}(undef, s...)

# For each column in model.distances (dealer order), pre-collect the corresponding
# CartesianIndex{4} entries in out (1 × T × J × F), combining the leading 1 with the
# 3-D product_lookup.f indices.  Computed once per model; eliminates names() allocation,
# product_lookup.f hash lookups, and idx.I splatting from the inner loop.
@memoize function μ_dist_plan(model)
    dist_cols = names(model.distances, 2)
    map(dist_cols) do uid
        idxs3 = get(model.product_lookup.f, uid, CartesianIndex{3}[])
        CartesianIndex{4}[CartesianIndex(1, idx.I...) for idx in idxs3]
    end
end

# Scratch buffer for per-dealer disutility values — sized to the number of dealers.
@memoize μ_disutil_prealloc(n) = Vector{Float32}(undef, n)

# demographic/product attribute interaction, demographic/price and random coeff interaction, distance disutility
function μ(params, model::AbstractDemandModel, zip_code)::SparseArray{Float32, 4}
    Πᵦ::Matrix{Float32} = reshape(
        @pull(demo_product_params, params, model),
        n_demographics(model),
        n_product_attributes(model),
    ) # (n_demo × n_product)
    Πₚ = @pull demo_price_params params model # (n_demo)
    η = @pull dist_params params model # (2)
    Π = hcat(Πᵦ, Πₚ) # n_demo x (n_product + 1)

    x_j = x_matrix_mu(model) # 1 × T × J × F × (n_product + price)
    D_i::Vector{Float32} = collect(model.agent_data[zip_code][DEMOGRAPHIC_ATTRIBUTES]) # n_demo-vector

    dist_f = @view model.distances[zip_code, :] # 1 x F

    # start with demographic utility over price and product attributes
    out = μ_prealloc(size(x_j)[1:(end - 1)])
    contract!(out, x_j, (1, 2, 3, 4, -1), SparseArray(Π'D_i), (-1,))
    # add distance disutility
    η1, η2 = η[1], η[2]
    f_idxs4 = μ_dist_plan(model)
    disutils = μ_disutil_prealloc(length(dist_f))
    @inbounds @simd for k in eachindex(disutils)
        disutils[k] = @fastmath exp(η1 + η2 * dist_f[k])
    end
    # regular getindex/setindex! dispatches to a method that can sometimes resize. Not thread safe!
    @inbounds for k in eachindex(f_idxs4)
        disutility = disutils[k]
        @simd for idx4 in f_idxs4[k]
            @inbounds out[idx4] -= disutility
        end
    end

    # add random utility (or zero for fixed coef model)
    # S × T × J × F
    ru = randomutility(params, model)
    S = size(ru, 1)

    if S > 1
        # Expand out to S dimensions and add random utility only to existing sparse entries
        expanded_out = similar(ru)

        # Copy existing entries and add random utility
        @inbounds for s in 1:S
            for (idx, val) in nonzero_pairs(out)
                (_, i...) = Tuple(idx)
                expanded_out[s, i...] = Float32(val)
            end
        end

        expanded_out += ru

        return expanded_out
    else
        return out
    end
end

const EMPTY_RCS = reshape(SparseArray([0.0f0]), ones(Int8, 4)...)
function randomutility(_, ::DemandModel{FixedCoefType})::SparseArray{Float32, 4}
    return EMPTY_RCS
end
function randomutility(params, model::DemandModel{RandomCoefType})::SparseArray{Float32, 4}
    σ = @pull random_coefs params model # (1) for price
    σ = abs(only(σ))
    X = x_matrix(model)
    p = similar(model.product_info, Float32)
    @inbounds for idx in nonzero_keys(model.product_info)
        p[idx] = X[Tuple(idx)..., end]
    end
    S = length(model.u.draws)
    out = SparseArray{Float32}(undef, (S, size(p)...))
    @inbounds for s in 1:S
        for (idx, val) in nonzero_pairs(p)
            out[s, Tuple(idx)...] = val * σ * model.u.draws[s]
        end
    end
    return out
end

@memoize grid_prealloc(s) = SparseArray{Float32}(undef, s...)

@memoize collected_zips(model) = collect(keys(model.counts_lookup.z))

# loglikelihood for all products, quarters, dealers
function loglikelihood(
    model::AbstractDemandModel,
    params::AbstractVector,
    z::String,
)::Float32
    mean_vec = SparseArray{Float32, 1}(ones(Float32, size(model.u.draws)))
    mean_vec /= Float32(size(model.u.draws, 1))

    ccps = CCPs(model, params, z) # S × T × J × F

    # Take mean across first dimension
    grid = grid_prealloc(size(ccps)[2:end])
    contract!(grid, mean_vec, (-1,), ccps, (-1, 2, 3, 4))

    map!(log, values(grid.data))

    # pull counts for zip
    iz = searchsortedfirst(collected_zips(model), z)
    counts = model.counts[iz, :, :, :]
    # log likelihood of all products, quarters, firms for zip
    return inner(grid, counts)
end

@memoize δ_reshape(s) = SparseArray{Float32}(undef, (1, s...))

function CCPs(model, params, z)
    # mean utilities (T × J × F)
    δ̂ = δ(params, model)
    δ_new = δ_reshape(size(δ̂))
    sizehint!(δ_new.data, nonzero_length(δ̂))
    @inbounds for (idx, val) in nonzero_pairs(δ̂)
        @inbounds δ_new[1, Tuple(idx)...] = val
    end
    δ̂ = δ_new
    # utility up to idiosyncratic error (S × T × J × F)
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
        add!(δ̂, μ̂)
    end

    # choice probabilities (S × T × J × F)
    CCPs!(model, grid)

    return grid
end

function CCPs!(model, grid)
    S = size(grid, 1)
    @inbounds for ts in values(model.product_lookup.t)
        for i in 1:S
            # outside good has utility 0; stabilize against max(0, max_inside)
            u = zero(Float32)
            for tup in ts
                @inbounds v = grid[i, tup.I...]
                v > u && (u = v)
            end
            outside_option = exp(-u)
            denom = outside_option
            for tup in ts
                e = exp(grid[i, tup.I...] - u)
                @inbounds grid[i, tup.I...] = e
                denom += e
            end
            inv_denom = 1.0f0 / denom
            for tup in ts
                @inbounds grid[i, tup.I...] *= inv_denom
            end
        end
    end
    return nothing
end

@memoize linear_params_prealloc(s) = SparseArray{Float32}(undef, s...)

function ξ̂(model::AbstractDemandModel, params::AbstractVector)::SparseArray{Float32, 3}
    # mean utilities (T × J × F)
    δ̂ = δ(params, model)
    residualize!(δ̂, model.product_lookup) # in-place residualization

    x_j = x_residualized(model) # T × J × F × (n_product + price)

    β = @pull mean_price_params params model # (1)

    # not decomposing for now
    # ρ = only(@pull rho params model)
    # γ = @pull group_fes params model # (n_group_fes)

    # Ω, i_Ω = Ω_matrix(model.ownership, model.adjacency)
    # @inbounds @simd for idx in i_Ω
    #     t, j, f, g = idx.I
    #     Ω[t, j, f, g] = model.ownership[t, f, g] + ρ * model.adjacency[t, j, f, g]
    # end

    contract!(
        linear_params_prealloc(size(x_j)[1:(end - 1)]), x_j, (1, 2, 3, -1), SparseArray(β), (-1,)
    )

    linear_params = linear_params_prealloc(size(x_j)[1:(end - 1)])

    # @inbounds for i in axes(Ω, 4)
    #     @info "Iteration $i of $(size(Ω, 4)) residualizing Ω matrix"
    #     linear_params += residualize!(Ω[:, :, :, i], model.product_lookup) * γ[i]
    # end

    return δ̂ - linear_params
end

function moments(model::AbstractDemandModel, params::AbstractVector)::AbstractVector
    ξ = ξ̂(model, params)
    Z = z_residualized(model)
    return mapreduce(vcat, axes(Z, ndims(Z))) do i_demo
        mapreduce(+, nonzero_pairs(ξ)) do (idx, val)
            @inbounds val * Z[Tuple(idx)..., i_demo]
        end
    end
end

function optimalW(model::AbstractDemandModel, params::AbstractVector)::AbstractMatrix
    ξ = ξ̂(model, params)
    Z = z_residualized(model)
    m = mapreduce(hcat, axes(Z, ndims(Z))) do i_demo
        [@inbounds val * Z[Tuple(idx)..., i_demo] for (idx, val) in nonzero_pairs(ξ)]
    end
    return Float32.(inv(cov(m)))
end

function Π̂(model::AbstractDemandModel, params::AbstractVector, W = I)::Float32
    m = moments(model, params)
    return 0.5f0 * only(m' * W * m)
end

function dΠ̂(model::AbstractDemandModel, params::AbstractVector, W = I)::AbstractVector
    m = moments(model, params)
    ∇m = reduce(hcat, (zeros(Float32, size(params)) for _ in 1:length(m)))

    i_δ = @indexof delta params model

    Z = z_matrix(model)

    # grad for delta
    @inbounds for (i, (_, idxs)) in enumerate(pairs(model.product_lookup.tjf))
        for col in axes(∇m, 2)
            row = i_δ[i]
            ∇m[row, col] = sum(Z[Tuple(idx)..., col] for idx in idxs)
        end
    end

    # grad for beta
    X = p_matrix(model)
    i_β = @indexof mean_price_params params model
    @inbounds for z_var in axes(Z, ndims(Z))
        @inbounds for x_var in axes(X, ndims(X))
            row = i_β[x_var]
            col = z_var
            ∇m[row, col] =
                -sum(
                    @inbounds X[Tuple(idx)..., x_var] * Z[Tuple(idx)..., z_var]
                    for idx in nonzero_keys(model.product_info)
                )
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

    grid = CCPs(model, params, z) # S × T × J × F

    # Take mean across first dimension
    contract!(
        grid_prealloc(size(grid)[2:end]), mean_vec, (-1,), grid, (-1, 2, 3, 4)
    )

    grid = grid_prealloc(size(grid)[2:end])

    # pull counts for zip
    iz = searchsortedfirst(collect(keys(model.counts_lookup.z)), z)
    counts = model.counts[iz, :, :, :]
    t_counts = SparseArray{Int64}(undef, size(counts, 1))
    @inbounds for (i, n) in nonzero_pairs(counts)
        t_counts[first(i.I)] += n
    end

    out = zeros(Float32, length(params))

    i_δ = @indexof delta params model

    # grad for delta
    @inbounds for (i, (_, idxs)) in enumerate(pairs(model.product_lookup.tjf))
        t = first(idxs).I[1]
        out[i_δ[i]] = sum(
            counts[idx] - t_counts[t] * grid[idx] for idx in idxs
        )
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
    @inbounds for i_X in axes(X, ndims(X))
        for i_D in axes(D_i, 1)
            row = i_Π[i_D, i_X]
            out[row] = sum(
                D_i[i_D] * X[Tuple(idx)..., i_X] *
                (counts[idx] - t_counts[first(idx.I)] * grid[idx])
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
            out[i_η[1]] -=
                exp(η[1] + η[2] * dist) *
                (counts[idx] - t_counts[first(idx.I)] * grid[idx])
            out[i_η[2]] -=
                dist * exp(η[1] + η[2] * dist) *
                (counts[idx] - t_counts[first(idx.I)] * grid[idx])
        end
    end

    return out
end
