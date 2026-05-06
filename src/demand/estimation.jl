@memoize δ_prealloc(model) = similar(model.product_info, Float32)
@memoize _resid_buf(model) = similar(model.product_info, Float32)
@memoize _t_counts_buf(model) = zeros(Int64, size(model.counts, 2))
@memoize _t_inside_buf(model) = zeros(Float32, size(model.counts, 2))

# Number of (t,j,f) product observations — denominator for mean moments.
@memoize n_obs(model) = Float32(nonzero_length(model.product_info))

# Aggregate observed purchase counts per δ group (sum over all zip codes).
# Used by the BLP contraction mapping for the inner δ optimization.
@memoize function δ_obs_counts(model)
    keys_plan, gidxs = δ_fill_plan(model)
    tjf_to_group = Dict(keys_plan[k] => Int(gidxs[k]) for k in eachindex(keys_plan))
    n_groups = length(model.product_lookup.tjf)
    counts = zeros(Float32, n_groups)
    for (idx, n) in nonzero_pairs(model.counts)
        _, t_i, j_i, f_i = idx.I
        g = get(tjf_to_group, CartesianIndex(t_i, j_i, f_i), 0)
        g > 0 && (counts[g] += Float32(n))
    end
    return counts
end

@memoize function _mean_vec(S::Int)
    v = SparseArray{Float32}(undef, (S,))
    inv_s = 1.0f0 / Float32(S)
    for i in 1:S
        v[i] = inv_s
    end
    return v
end

@memoize _demo_zip(model, z::String) =
    collect(model.agent_data[z][DEMOGRAPHIC_ATTRIBUTES])

# Sum of instruments over each (t,j,f) product group — rows of ∂m/∂δ; model-fixed.
@memoize function _Z_group_sums(model)
    Z = z_matrix(model)
    n_z = size(Z, ndims(Z))
    n_groups = length(model.product_lookup.tjf)
    sums = zeros(Float32, n_groups, n_z)
    @inbounds for (i, (_, idxs)) in enumerate(pairs(model.product_lookup.tjf))
        for col in 1:n_z
            sums[i, col] = sum(Z[Tuple(idx)..., col] for idx in idxs)
        end
    end
    return sums
end

# X'Z cross-product for the price-attribute block of ∂m/∂β; model-fixed.
@memoize function _XZ_cross(model)
    X = p_matrix(model)
    Z = z_matrix(model)
    n_x = size(X, ndims(X))
    n_z = size(Z, ndims(Z))
    XZ = zeros(Float32, n_x, n_z)
    @inbounds for idx in nonzero_keys(model.product_info)
        t = Tuple(idx)
        for i_x in 1:n_x, i_z in 1:n_z
            XZ[i_x, i_z] += X[t..., i_x] * Z[t..., i_z]
        end
    end
    return XZ
end

@memoize _∇m_buf(n_params::Int, n_moments::Int) = Matrix{Float32}(undef, n_params, n_moments)

@memoize function _counts_zip(model, z::String)
    iz = searchsortedfirst(collected_zips(model), z)
    return model.counts[iz, :, :, :]
end

@memoize function _dist_zip_plan(model, z::String)
    col_names = names(model.distances, 2)
    dist_row = model.distances[z, :]
    result = Tuple{Float32, Vector{CartesianIndex{3}}}[]
    for (uid, dist) in zip(col_names, dist_row)
        idxs = get(model.product_lookup.f, uid, CartesianIndex{3}[])
        isempty(idxs) && continue
        push!(result, (Float32(dist), idxs))
    end
    return result
end

@memoize function δ_fill_plan(model)
    n = sum(length(idxs) for idxs in values(model.product_lookup.tjf))
    keys_plan = Vector{CartesianIndex{3}}(undef, n)
    gidxs = Vector{Int32}(undef, n)
    k = 1
    for (i, idxs) in enumerate(values(model.product_lookup.tjf))
        for idx in idxs
            keys_plan[k] = idx
            gidxs[k] = Int32(i)
            k += 1
        end
    end
    return keys_plan, gidxs
end

@memoize δ_indices(model) = first(δ_fill_plan(model))

function δ(params::AbstractVector, model::DemandModel)::SparseArray{Float32, 3}
    guesses::Vector{Float32} = @pull delta params model # one delta per (t, j_trim, f)
    deltas = δ_prealloc(model)
    keys_plan, gidxs = δ_fill_plan(model)
    h = deltas.data
    @inbounds for k in eachindex(keys_plan)
        h[keys_plan[k]] = guesses[gidxs[k]]
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
    @inbounds @threads for k in eachindex(f_idxs4)
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
    S = length(model.u.draws)
    ccps = CCPs(model, params, z) # S × T × J × F

    # Take mean across first dimension
    grid = grid_prealloc(size(ccps)[2:end])
    contract!(grid, _mean_vec(S), (-1,), ccps, (-1, 2, 3, 4))

    # Outside-good contribution: (M_z - obs_t) * log(1 - Σ_{j,f} CCP_{tjf})
    # Must be computed before map!(log, ...) mutates grid.
    M_z = model.agent_data[z].M
    counts = _counts_zip(model, z)
    t_counts = _t_counts_buf(model)
    fill!(t_counts, zero(Int64))
    @inbounds for (i, n) in nonzero_pairs(counts)
        t_counts[first(i.I)] += n
    end
    t_inside = _t_inside_buf(model)
    fill!(t_inside, 0.0f0)
    @inbounds for idx in nonzero_keys(model.product_info)
        t_inside[first(idx.I)] += grid[idx]
    end
    ll_outside = sum(enumerate(t_counts)) do (t, n_obs)
        n_out = M_z - Float32(n_obs)
        n_out > 0.0f0 ? n_out * log(max(1.0f0 - t_inside[t], 1.0f-10)) : 0.0f0
    end

    map!(log, values(grid.data))
    return inner(grid, counts) + ll_outside
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
    N = n_obs(model)
    return mapreduce(vcat, axes(Z, ndims(Z))) do i_demo
        mapreduce(+, nonzero_pairs(ξ)) do (idx, val)
            @inbounds val * Z[Tuple(idx)..., i_demo]
        end
    end ./ N
end

function optimalW(model::AbstractDemandModel, params::AbstractVector)::AbstractMatrix
    ξ = ξ̂(model, params)
    Z = z_residualized(model)
    m = mapreduce(hcat, axes(Z, ndims(Z))) do i_demo
        [@inbounds val * Z[Tuple(idx)..., i_demo] for (idx, val) in nonzero_pairs(ξ)]
    end
    C = cov(m)
    W = pinv(C; rtol = 1e-4)
    @info "optimalW: cond(cov)=$(round(cond(C); sigdigits=4)) max(W)=$(round(maximum(abs, W); sigdigits=4))"
    return Float32.(W)
end

function Π̂(model::AbstractDemandModel, params::AbstractVector, W = I)::Float32
    m = moments(model, params)
    return 0.5f0 * only(m' * W * m)
end

function dΠ̂(model::AbstractDemandModel, params::AbstractVector, W = I)::AbstractVector
    m = moments(model, params)

    N = n_obs(model)
    Z_gsums = _Z_group_sums(model)  # n_groups × n_z, model-fixed
    XZ = _XZ_cross(model)           # n_x × n_z, model-fixed

    ∇m = _∇m_buf(length(params), size(Z_gsums, 2))
    fill!(∇m, zero(Float32))

    i_δ = @indexof delta params model
    @inbounds for (i, _) in enumerate(pairs(model.product_lookup.tjf))
        ∇m[i_δ[i], :] .= @view(Z_gsums[i, :]) ./ N
    end

    i_β = @indexof mean_price_params params model
    @inbounds for x_var in axes(XZ, 1)
        ∇m[i_β[x_var], :] .= .-@view(XZ[x_var, :]) ./ N
    end

    return 0.5f0 * ∇m * W * m
end

function dloglikelihood(
    model::AbstractDemandModel,
    params::AbstractVector,
    z::String,
)::AbstractVector
    S = length(model.u.draws)
    grid_S = CCPs(model, params, z) # S × T × J × F

    # Average CCPs over draws into T × J × F buffer
    avg_grid = grid_prealloc(size(grid_S)[2:end])
    contract!(avg_grid, _mean_vec(S), (-1,), grid_S, (-1, 2, 3, 4))

    counts = _counts_zip(model, z)
    M_z = model.agent_data[z].M

    # residual = obs - M_z * CCP; gradient of full logL including outside good.
    # The outside-good term (M_z - obs_t)*log(1-Σ CCP) contributes -M_z*CCP_j to ∂logL/∂δ_j,
    # which cancels the obs_t*CCP_j cross-term, leaving obs_j - M_z*CCP_j exactly.
    resid = _resid_buf(model)
    @inbounds for idx in nonzero_keys(model.product_info)
        resid[idx] = counts[idx] - M_z * avg_grid[idx]
    end

    out = zeros(Float32, length(params))
    i_δ = @indexof delta params model

    # grad for delta
    @inbounds for (i, (_, idxs)) in enumerate(pairs(model.product_lookup.tjf))
        out[i_δ[i]] = sum(resid[idx] for idx in idxs)
    end

    # grad for demo_product and demo_price params
    X = x_matrix(model)
    i_Πᵦ = reshape(
        @indexof(demo_product_params, params, model),
        n_demographics(model),
        n_product_attributes(model),
    )
    i_Πₚ = @indexof demo_price_params params model
    i_Π = hcat(i_Πᵦ, i_Πₚ)
    D_i = _demo_zip(model, z)
    @inbounds for i_X in axes(X, ndims(X))
        # inner product of X[:, i_X] with resid — same for all demographics
        s = sum(X[Tuple(idx)..., i_X] * resid[idx] for idx in nonzero_keys(model.product_info))
        for i_D in axes(D_i, 1)
            out[i_Π[i_D, i_X]] = D_i[i_D] * s
        end
    end

    # grad for distance params
    η = @pull dist_params params model
    i_η = @indexof dist_params params model
    @inbounds for (dist, idxs) in _dist_zip_plan(model, z)
        e = @fastmath exp(η[1] + η[2] * dist)
        s = sum(resid[idx] for idx in idxs)
        out[i_η[1]] -= e * s
        out[i_η[2]] -= dist * e * s
    end

    return out
end
