# Dense per-block evaluation path for the discrete-choice CCP model.
#
# Same math and parameter ordering as estimation.jl (the SparseArrayKit-backed path).
# The key structural observation: the (t, j, f) sparsity pattern in product_info is
# the same across all zips — δ is shared and the only zip-varying quantities are
# demographics D[z] (4-vec) and distance dist[z, :] (F-vec). So for each time block
# we can stack all zips into a dense N_t × Z matrix and softmax per column with BLAS.
#
# Memory at full Z=10K, N≈100K: ~16 GB total
#   DenseModel static: dist_blocks 4 GB, counts (sparse) <100 MB, X/D/M small
#   DenseWork per-eval: U_blocks 4 GB, E_blocks 4 GB, resid_blocks 4 GB, G_π/α small
# For laptops or memory-constrained nodes, batch the zip list and accumulate.

module Dense

using LinearAlgebra
using SparseArrays
using Base.Threads: @threads

import ..DemandModel, ..FixedCoefType, ..AbstractDemandModel
import ..PRODUCT_ATTRIBUTES, ..DEMOGRAPHIC_ATTRIBUTES
import ..n_delta, ..n_demographics, ..n_product_attributes
import ..n_demo_product_params, ..n_demo_price_params, ..n_dist_params
import ..n_random_coefs, ..total_n_params
import ..δ_indices, ..δ_t_blocks, ..collected_zips
import ..ORDER
import SparseArrayKit
import ..@indexof

export DenseModel, DenseWork,
    forward_pass!, eval_ll!, eval_ll_δgrad!, eval_combined!,
    woodbury_newton_step!, param_indices

# -------------------------------------------------------------------------------------------------
# Parameter-vector index ranges — delegate to @indexof so ORDER changes propagate here.
# -------------------------------------------------------------------------------------------------

function param_indices(model::AbstractDemandModel)
    # @indexof ignores its second argument (it's only used by @pull); pass nothing.
    i_δ = @indexof delta nothing model
    i_Πᵦ = @indexof demo_product_params nothing model
    i_Πp = @indexof demo_price_params nothing model
    i_η = @indexof dist_params nothing model
    return (; i_δ, i_Πᵦ, i_Πp, i_η)
end

# -------------------------------------------------------------------------------------------------
# DenseModel: read-only precomputed indexing for the per-block dense layout.
# -------------------------------------------------------------------------------------------------

"""
DenseModel(model, zips)

Precomputes per-block dense data for evaluation across all `zips` simultaneously.

Field shapes (N = n_delta, Z = length(zips), K = n_product_attributes, Dn = n_demographics):

  - `block_param_idxs[bi]::Vector{Int32}` (length N_bi): block-local row → param index in δ
  - `X_blocks[bi]::Matrix{Float32}` ((K+1) × N_bi): product attributes + price, gathered per block
  - `dist_blocks[bi]::Matrix{Float32}` (N_bi × Z): standardized distance dist[z, k_to_f[k]]
  - `counts_blocks[bi]::SparseMatrixCSC{Int32,Int32}` (N_bi × Z): purchase counts
  - `t_obs::Matrix{Int32}` (n_blocks × Z): total inside-good purchases per (block, zip)
  - `D_dense::Matrix{Float32}` (Dn × Z): standardized demographics per zip
  - `M_zs::Vector{Float32}` (Z): market sizes
"""
struct DenseModel{T <: AbstractDemandModel}
    model::T
    N::Int
    Z::Int
    K::Int
    Dn::Int
    n_blocks::Int
    zips::Vector{String}
    block_param_idxs::Vector{Vector{Int32}}
    X_blocks::Vector{Matrix{Float32}}
    dist_blocks::Vector{Matrix{Float32}}
    counts_blocks::Vector{SparseMatrixCSC{Int32, Int32}}
    t_obs::Matrix{Int32}
    D_dense::Matrix{Float32}
    M_zs::Vector{Float32}
end

function DenseModel(model::DemandModel{FixedCoefType}, zips::Vector{String})
    Z = length(zips)
    N = n_delta(model)
    K = n_product_attributes(model)
    Dn = n_demographics(model)

    all_idxs = δ_indices(model)                  # Vector{CartesianIndex{3}}, length N
    @assert length(all_idxs) == N

    blocks_dict = δ_t_blocks(model)
    block_keys = sort(collect(keys(blocks_dict)))
    n_blocks = length(block_keys)
    block_param_idxs = [Int32.(blocks_dict[k]) for k in block_keys]

    # X_full: (K+1) × N — product attributes (Bool→Float32) + standardized price
    X_full = Matrix{Float32}(undef, K + 1, N)
    @inbounds for k in 1:N
        info = model.product_info[all_idxs[k]]
        for (p_i, attr) in enumerate(PRODUCT_ATTRIBUTES)
            X_full[p_i, k] = Float32(getfield(info, attr))
        end
        X_full[K + 1, k] = info.p
    end
    X_blocks = [X_full[:, block_param_idxs[bi]] for bi in 1:n_blocks]

    # k → dealer column in model.distances
    dealer_uids = names(model.distances, 2)
    dealer_to_col = Dict(u => Int32(i) for (i, u) in enumerate(dealer_uids))
    tjf_keys = collect(keys(model.product_lookup.tjf))
    @assert length(tjf_keys) == N
    k_to_f = Int32[dealer_to_col[k_tup[3]] for k_tup in tjf_keys]

    # D_dense (Dn × Z) — column order matches `zips`
    D_dense = Matrix{Float32}(undef, Dn, Z)
    @inbounds for (zi, z) in enumerate(zips)
        ai = model.agent_data[z]
        for (di, attr) in enumerate(DEMOGRAPHIC_ATTRIBUTES)
            D_dense[di, zi] = Float32(getfield(ai, attr))
        end
    end

    # dist_zf (Z × F): one row per zip in our ordering, dense access
    F = size(model.distances, 2)
    dist_zf = Matrix{Float32}(undef, Z, F)
    @inbounds for (zi, z) in enumerate(zips)
        row = model.distances[z, :]
        for f in 1:F
            dist_zf[zi, f] = Float32(row[f])
        end
    end

    # dist_blocks[bi] (N_bi × Z): gather dist_zf[z, k_to_f[k]] in k-major form
    dist_blocks = Vector{Matrix{Float32}}(undef, n_blocks)
    for bi in 1:n_blocks
        param_idxs = block_param_idxs[bi]
        N_bi = length(param_idxs)
        M = Matrix{Float32}(undef, N_bi, Z)
        @inbounds for k in 1:N_bi
            f_col = k_to_f[param_idxs[k]]
            for z in 1:Z
                M[k, z] = dist_zf[z, f_col]
            end
        end
        dist_blocks[bi] = M
    end

    M_zs = Float32[model.agent_data[z].M for z in zips]

    # counts: from model.counts (z, t, j, f) → per-block SparseMatrixCSC of N_bi × Z
    tjf_to_block_k = Dict{CartesianIndex{3}, Tuple{Int32, Int32}}()
    for bi in 1:n_blocks
        for (k_local, p) in enumerate(block_param_idxs[bi])
            tjf_to_block_k[all_idxs[p]] = (Int32(bi), Int32(k_local))
        end
    end
    target_z_col = Dict(z => Int32(i) for (i, z) in enumerate(zips))
    z_dim_to_str = collected_zips(model)

    I_pb = [Int32[] for _ in 1:n_blocks]
    J_pb = [Int32[] for _ in 1:n_blocks]
    V_pb = [Int32[] for _ in 1:n_blocks]
    for idx4 in SparseArrayKit.nonzero_keys(model.counts)
        zi, ti, ji, fi = idx4.I
        z_str = z_dim_to_str[zi]
        col = get(target_z_col, z_str, Int32(0))
        col == 0 && continue
        bk = get(tjf_to_block_k, CartesianIndex(ti, ji, fi), nothing)
        isnothing(bk) && continue
        bi, k_local = bk
        push!(I_pb[bi], k_local)
        push!(J_pb[bi], col)
        push!(V_pb[bi], Int32(model.counts[idx4]))
    end
    counts_blocks = SparseMatrixCSC{Int32, Int32}[
        sparse(I_pb[bi], J_pb[bi], V_pb[bi], length(block_param_idxs[bi]), Z) for bi in 1:n_blocks
    ]

    # t_obs[bi, z] = column sums of counts_blocks[bi]
    t_obs = zeros(Int32, n_blocks, Z)
    @inbounds for bi in 1:n_blocks
        cb = counts_blocks[bi]
        for col in 1:Z
            for nz in cb.colptr[col]:(cb.colptr[col + 1] - 1)
                t_obs[bi, col] += cb.nzval[nz]
            end
        end
    end

    return DenseModel(
        model, N, Z, K, Dn, n_blocks, zips,
        block_param_idxs, X_blocks, dist_blocks, counts_blocks, t_obs, D_dense, M_zs,
    )
end

# -------------------------------------------------------------------------------------------------
# DenseWork: mutable per-evaluation buffers. Preallocate once, reuse across calls.
# -------------------------------------------------------------------------------------------------

struct DenseWork
    U_blocks::Vector{Matrix{Float32}}    # CCPs (or utility pre-softmax) per block, N_bi × Z
    E_blocks::Vector{Matrix{Float32}}    # exp(η₁ + η₂·dist) per block, N_bi × Z
    resid_blocks::Vector{Matrix{Float32}} # counts - M_z·CCP per block, N_bi × Z
    inside_sum::Matrix{Float32}          # Σ_{k∈bi} CCP[k,z], n_blocks × Z
    α::Matrix{Float32}                    # Σ_bi X_bi · resid_bi, (K+1) × Z
    G_π::Matrix{Float32}                  # per-zip score wrt π, N_NONδ × Z
    score_η1::Vector{Float32}            # per-zip η₁ score (last two rows of G_π)
    score_η2::Vector{Float32}
end

function DenseWork(dm::DenseModel)
    U_blocks = [zeros(Float32, length(dm.block_param_idxs[bi]), dm.Z) for bi in 1:(dm.n_blocks)]
    E_blocks = [zeros(Float32, length(dm.block_param_idxs[bi]), dm.Z) for bi in 1:(dm.n_blocks)]
    resid_blocks = [zeros(Float32, length(dm.block_param_idxs[bi]), dm.Z) for bi in 1:(dm.n_blocks)]
    inside_sum = zeros(Float32, dm.n_blocks, dm.Z)
    N_NONδ = (dm.K + 1) * dm.Dn + 2  # Πᵦ (Dn × K) + Πp (Dn) + η (2)
    α = zeros(Float32, dm.K + 1, dm.Z)
    G_π = zeros(Float32, N_NONδ, dm.Z)
    score_η1 = zeros(Float32, dm.Z)
    score_η2 = zeros(Float32, dm.Z)
    return DenseWork(U_blocks, E_blocks, resid_blocks, inside_sum, α, G_π, score_η1, score_η2)
end

# -------------------------------------------------------------------------------------------------
# Forward pass: utility → CCP per block.
# -------------------------------------------------------------------------------------------------

# Per-column softmax over a block's rows, with outside-good utility = 0 (stabilized).
# Writes inside_sum[z] = Σ_k CCP[k,z].
@inline function softmax_block!(U::AbstractMatrix{Float32}, inside_sum_row::AbstractVector{Float32})
    N_t, Z = size(U)
    @inbounds for z in 1:Z
        u_max = 0.0f0
        for k in 1:N_t
            v = U[k, z]
            v > u_max && (u_max = v)
        end
        outside = exp(-u_max)
        denom = outside
        for k in 1:N_t
            e = exp(U[k, z] - u_max)
            U[k, z] = e
            denom += e
        end
        inv_denom = 1.0f0 / denom
        sm_sum = 0.0f0
        for k in 1:N_t
            U[k, z] *= inv_denom
            sm_sum += U[k, z]
        end
        inside_sum_row[z] = sm_sum
    end
end

"""
Build Π'·D matrix (K+1 × Z) where Π = [Πᵦ | Πp].

Top K rows are Πᵦ' · D; last row is Πp' · D. Used as the "shared per-zip coefficient" vector
that multiplies each block's X to give the demographic utility contribution.
"""
function compute_ΠtD!(
    ΠtD::Matrix{Float32}, Πᵦ::AbstractMatrix{Float32}, Πp::AbstractVector{Float32},
    D::Matrix{Float32},
)
    _, K = size(Πᵦ)  # Πᵦ is Dn × K → we want Πᵦ' · D which is K × Z
    mul!(view(ΠtD, 1:K, :), Πᵦ', D)
    Z = size(D, 2)
    @inbounds for z in 1:Z
        s = 0.0f0
        for d in 1:size(D, 1)
            s += Πp[d] * D[d, z]
        end
        ΠtD[K + 1, z] = s
    end
end

"""
forward_pass!(work, dm, params)

Computes per-block CCPs into `work.U_blocks` and exp distance utilities into `work.E_blocks`,
along with `work.inside_sum[bi, z] = Σ_{k∈block bi} CCP[k, z]`.

Parameters are read via `param_indices(dm.model)`.
Multi-threaded across time blocks (independent work per `bi`).
"""
function forward_pass!(work::DenseWork, dm::DenseModel, params::AbstractVector{Float32})
    idx = param_indices(dm.model)
    δ = view(params, idx.i_δ)
    Πᵦ = reshape(view(params, idx.i_Πᵦ), dm.Dn, dm.K)
    Πp = view(params, idx.i_Πp)
    η = view(params, idx.i_η)
    η1, η2 = η[1], η[2]

    ΠtD = Matrix{Float32}(undef, dm.K + 1, dm.Z)
    compute_ΠtD!(ΠtD, Πᵦ, Πp, dm.D_dense)

    @threads :static for bi in 1:(dm.n_blocks)
        U_bi = work.U_blocks[bi]
        E_bi = work.E_blocks[bi]
        X_bi = dm.X_blocks[bi]
        dist_bi = dm.dist_blocks[bi]
        param_idxs = dm.block_param_idxs[bi]
        N_bi = size(U_bi, 1)

        # U = X_bi' * ΠtD   →   N_bi × Z (one BLAS GEMM)
        mul!(U_bi, X_bi', ΠtD)

        # Add δ and subtract exp distance disutility; store E for later use
        @inbounds for z in 1:(dm.Z)
            for k in 1:N_bi
                e = @fastmath exp(η1 + η2 * dist_bi[k, z])
                E_bi[k, z] = e
                U_bi[k, z] += δ[param_idxs[k]] - e
            end
        end

        softmax_block!(U_bi, view(work.inside_sum, bi, :))
    end
    return nothing
end

# Inside-good ll for one block given counts (sparse) and CCPs (after softmax in U).
# Computes Σ_{(k,z) ∈ nnz(counts)} counts[k,z] * log(max(CCP[k,z], ε)).
function _inside_ll_block(U::Matrix{Float32}, counts::SparseMatrixCSC{Int32, Int32})
    s = 0.0f0
    @inbounds for col in 1:size(counts, 2)
        for nz in counts.colptr[col]:(counts.colptr[col + 1] - 1)
            row = counts.rowval[nz]
            val = Float32(counts.nzval[nz])
            s += val * log(max(U[row, col], 1.0f-30))
        end
    end
    return s
end

# Outside-good ll: Σ_{bi, z} (M_z - t_obs[bi,z]) · log(max(1 - inside_sum[bi,z], ε))
function _outside_ll(work::DenseWork, dm::DenseModel)
    ll = 0.0f0
    @inbounds for z in 1:(dm.Z)
        for bi in 1:(dm.n_blocks)
            n_out = dm.M_zs[z] - Float32(dm.t_obs[bi, z])
            if n_out > 0.0f0
                p_out = max(1.0f0 - work.inside_sum[bi, z], 1.0f-10)
                ll += n_out * log(p_out)
            end
        end
    end
    return ll
end

"""
eval_ll!(work, dm, params) → ll::Float32

Runs forward pass and returns the log-likelihood. Cheaper than `eval_combined!` —
no resid, no gradient, no BHHH. Used by the inner-Newton line search.
"""
function eval_ll!(work::DenseWork, dm::DenseModel, params::AbstractVector{Float32})
    forward_pass!(work, dm, params)
    ll_in = 0.0f0
    for bi in 1:(dm.n_blocks)
        ll_in += _inside_ll_block(work.U_blocks[bi], dm.counts_blocks[bi])
    end
    return ll_in + _outside_ll(work, dm)
end

"""
eval_ll_δgrad!(work, dm, params; δ_grad) → ll::Float32

Runs forward pass, returns ll, and fills `δ_grad[1:N] = ∂ℓ/∂δ`. Cheaper than
`eval_combined!` — skips Π and η gradient assembly and BHHH. Used by line-search dϕ.
"""
function eval_ll_δgrad!(
    work::DenseWork, dm::DenseModel, params::AbstractVector{Float32};
    δ_grad::AbstractVector{Float32},
)
    forward_pass!(work, dm, params)
    fill!(δ_grad, 0.0f0)
    ll_in = 0.0f0
    @inbounds for bi in 1:(dm.n_blocks)
        U_bi = work.U_blocks[bi]
        resid_bi = work.resid_blocks[bi]
        counts_bi = dm.counts_blocks[bi]
        param_idxs = dm.block_param_idxs[bi]
        N_bi = size(U_bi, 1)
        # resid_bi = -M_z · CCP
        for z in 1:(dm.Z)
            mz = dm.M_zs[z]
            for k in 1:N_bi
                resid_bi[k, z] = -mz * U_bi[k, z]
            end
        end
        # add counts and accumulate inside ll
        ll_block = 0.0f0
        for col in 1:(dm.Z)
            for nz in counts_bi.colptr[col]:(counts_bi.colptr[col + 1] - 1)
                row = counts_bi.rowval[nz]
                val = Float32(counts_bi.nzval[nz])
                resid_bi[row, col] += val
                ll_block += val * log(max(U_bi[row, col], 1.0f-30))
            end
        end
        ll_in += ll_block
        # δ_grad: row sums
        for k in 1:N_bi
            s = 0.0f0
            for z in 1:(dm.Z)
                s += resid_bi[k, z]
            end
            δ_grad[param_idxs[k]] = s
        end
    end
    return ll_in + _outside_ll(work, dm)
end

# -------------------------------------------------------------------------------------------------
# Full combined evaluation: ll + gradient (all params) + BHHH (non-δ).
# -------------------------------------------------------------------------------------------------

"""
eval_combined!(work, dm, params; grad, bhhh) → ll::Float32

Single pass: forward, ll, full gradient (∂ℓ/∂params), BHHH (∂ℓ/∂π outer products).

Layout:

  - `grad[i_δ]`  = Σ_z (counts[k,z] - M_z·CCP[k,z])           # δ ascent gradient of ℓ
  - `grad[i_Πᵦ]` = (D · α[1:K,:]')[d,p]                        # column-major (d fastest)
  - `grad[i_Πp]` = D · α[K+1,:]'                                # Dn-vector
  - `grad[i_η]`  = -(sum E·resid, sum dist·E·resid)             # 2-vector
  - `bhhh`       = G_π · G_π'                                   # N_NONδ × N_NONδ, symmetric

`work.U_blocks` still holds CCPs on exit (needed by Woodbury Newton step).
"""
function eval_combined!(
    work::DenseWork, dm::DenseModel, params::AbstractVector{Float32};
    grad::AbstractVector{Float32}, bhhh::AbstractMatrix{Float32},
)
    forward_pass!(work, dm, params)

    idx = param_indices(dm.model)
    fill!(grad, 0.0f0)
    fill!(work.α, 0.0f0)
    fill!(work.score_η1, 0.0f0)
    fill!(work.score_η2, 0.0f0)

    ll_inside = 0.0f0
    # Per-block: build resid, fill δ grad, accumulate α (Π contributions) and per-zip η scores
    for bi in 1:(dm.n_blocks)
        U_bi = work.U_blocks[bi]
        E_bi = work.E_blocks[bi]
        resid_bi = work.resid_blocks[bi]
        X_bi = dm.X_blocks[bi]
        dist_bi = dm.dist_blocks[bi]
        counts_bi = dm.counts_blocks[bi]
        param_idxs = dm.block_param_idxs[bi]
        N_bi = size(U_bi, 1)

        # resid = -M_z · CCP
        @inbounds for z in 1:(dm.Z)
            mz = dm.M_zs[z]
            for k in 1:N_bi
                resid_bi[k, z] = -mz * U_bi[k, z]
            end
        end
        # add counts and inside ll
        ll_block = 0.0f0
        @inbounds for col in 1:(dm.Z)
            for nz in counts_bi.colptr[col]:(counts_bi.colptr[col + 1] - 1)
                row = counts_bi.rowval[nz]
                val = Float32(counts_bi.nzval[nz])
                resid_bi[row, col] += val
                ll_block += val * log(max(U_bi[row, col], 1.0f-30))
            end
        end
        ll_inside += ll_block

        # δ grad: row sums of resid_bi
        @inbounds for k in 1:N_bi
            s = 0.0f0
            for z in 1:(dm.Z)
                s += resid_bi[k, z]
            end
            grad[param_idxs[k]] = s
        end

        # α += X_bi · resid_bi   ((K+1) × N_bi · N_bi × Z → (K+1) × Z)
        BLAS.gemm!('N', 'N', 1.0f0, X_bi, resid_bi, 1.0f0, work.α)

        # η: per-zip column sums (s1, s2) and global sums
        @inbounds for z in 1:(dm.Z)
            s1 = 0.0f0
            s2 = 0.0f0
            for k in 1:N_bi
                er = E_bi[k, z] * resid_bi[k, z]
                s1 += er
                s2 += dist_bi[k, z] * er
            end
            work.score_η1[z] -= s1
            work.score_η2[z] -= s2
        end
    end

    ll = ll_inside + _outside_ll(work, dm)

    # Π gradient: D · α' is (Dn × (K+1)), giving ∂ℓ/∂Π[d,p] in (d,p) order
    Π_grad = Matrix{Float32}(undef, dm.Dn, dm.K + 1)
    mul!(Π_grad, dm.D_dense, work.α')

    # Πᵦ layout: column-major over (d, p) with d varying fastest (matches @indexof in models.jl)
    i_Πᵦ_start = first(idx.i_Πᵦ)
    @inbounds for p in 1:(dm.K), d in 1:(dm.Dn)
        grad[i_Πᵦ_start + (p - 1) * dm.Dn + (d - 1)] = Π_grad[d, p]
    end
    i_Πp_start = first(idx.i_Πp)
    @inbounds for d in 1:(dm.Dn)
        grad[i_Πp_start + (d - 1)] = Π_grad[d, dm.K + 1]
    end

    # η gradient (global sums of per-zip scores)
    i_η_start = first(idx.i_η)
    grad[i_η_start] = sum(work.score_η1)
    grad[i_η_start + 1] = sum(work.score_η2)

    # Per-zip score G_π (N_NONδ × Z) for BHHH
    N_NONδ = size(work.G_π, 1)
    @inbounds for p in 1:(dm.K), d in 1:(dm.Dn)
        row = (p - 1) * dm.Dn + d
        for z in 1:(dm.Z)
            work.G_π[row, z] = dm.D_dense[d, z] * work.α[p, z]
        end
    end
    @inbounds for d in 1:(dm.Dn)
        row = dm.K * dm.Dn + d
        for z in 1:(dm.Z)
            work.G_π[row, z] = dm.D_dense[d, z] * work.α[dm.K + 1, z]
        end
    end
    @inbounds for z in 1:(dm.Z)
        work.G_π[N_NONδ - 1, z] = work.score_η1[z]
        work.G_π[N_NONδ, z] = work.score_η2[z]
    end

    # BHHH = G_π · G_π'  (single SYRK, then symmetrize)
    fill!(bhhh, 0.0f0)
    BLAS.syrk!('U', 'N', 1.0f0, work.G_π, 0.0f0, bhhh)
    @inbounds for j in 1:N_NONδ, i in (j + 1):N_NONδ
        bhhh[i, j] = bhhh[j, i]
    end

    return ll
end

# -------------------------------------------------------------------------------------------------
# Woodbury Newton step over δ. Same math as scripts/demand_estimation.jl, operating on
# work.U_blocks (CCPs from the latest combined/forward call) and a δ-gradient vector.
# -------------------------------------------------------------------------------------------------

# Returns (-H_t)^{-1} g_t — Newton ascent direction for ℓ over δ_t.
# Falls back to g_t on Float32 SingularException in the inner Cholesky (degenerate CCPs).
function _woodbury_block(
    U::Matrix{Float32}, M_z::Vector{Float32}, g_t::Vector{Float32}
)::Vector{Float32}
    Z = length(M_z)
    d = U * M_z
    inv_d = 1.0f0 ./ max.(d, 1.0f-12)
    sqrt_inv_d = sqrt.(inv_d)
    A = sqrt_inv_d .* U
    v = inv_d .* g_t
    middle = zeros(Float32, Z, Z)
    middle[diagind(middle)] .= 1.0f0 ./ M_z
    BLAS.syrk!('U', 'T', -1.0f0, A, 1.0f0, middle)
    rhs = A' * (sqrt_inv_d .* g_t)
    try
        w = Symmetric(middle, :U) \ rhs
        return v .+ sqrt_inv_d .* (A * w)
    catch e
        e isa SingularException || rethrow(e)
        return g_t
    end
end

"""
woodbury_newton_step!(step, work, dm, g_δ)

Writes Newton ascent direction over δ (length N) into `step`, using `work.U_blocks`
(CCPs from latest forward pass) and the per-block market sizes.
Threaded across time blocks.
"""
function woodbury_newton_step!(
    step::AbstractVector{Float32}, work::DenseWork, dm::DenseModel, g_δ::AbstractVector{Float32}
)
    @threads :static for bi in 1:(dm.n_blocks)
        param_idxs = dm.block_param_idxs[bi]
        g_t = Float32[g_δ[k] for k in param_idxs]
        d_t = _woodbury_block(work.U_blocks[bi], dm.M_zs, g_t)
        @inbounds for (k_local, p) in enumerate(param_idxs)
            step[p] = d_t[k_local]
        end
    end
    return step
end

end # module Dense
