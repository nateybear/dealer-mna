# Dense per-block evaluation path for the discrete-choice CCP model.
#
# The (t, j, f) layout is fully baked into DemandModel: δ is shared across zips, and the
# only zip-varying quantities are demographics D[z] (4-vec) and distance dist[z, :] (F-vec).
# For each time block, all zips are stacked into a dense N_bi × Z matrix and softmax is
# computed per column with BLAS.

export DemandWork, Counterfactual,
    forward_pass!, eval_ll!, eval_ll_δgrad!, eval_combined!,
    woodbury_newton_step!, eval_envelope_hessian!, definiteness_shift!,
    param_indices, build_nonδ_labels,
    AbstractTweak, PriceMultiply, PriceChange, ModelAdd, ModelDelete,
    apply!, undo!, refresh!,
    share_matrix, expected_demand, foc_residual, dealer_block_locals

using FixedEffectModels
import ..pluck

# -------------------------------------------------------------------------------------------------
# Parameter indexing — delegates to @indexof so ORDER changes propagate automatically.
# -------------------------------------------------------------------------------------------------

function param_indices(model::DemandModel)
    i_δ = @indexof delta nothing model
    i_Πᵦ = @indexof demo_product_params nothing model
    i_Πp = @indexof demo_price_params nothing model
    i_η = @indexof dist_params nothing model
    return (; i_δ, i_Πᵦ, i_Πp, i_η)
end

# Labels for the non-δ parameter block (length N_NONδ = K*Dn + Dn + 2 = 22 at K=4, Dn=4).
function build_nonδ_labels()
    D = string.(DEMOGRAPHIC_ATTRIBUTES)
    P = string.(PRODUCT_ATTRIBUTES)
    return vcat(
        ["Π[$d,$p]" for p in P for d in D],
        ["Πp[$d]" for d in D],
        ["η_baseline", "η_dist"],
    )
end

# -------------------------------------------------------------------------------------------------
# DemandWork: per-evaluation buffers (preallocated once, reused). Combines the former
# DenseWork (forward-pass buffers) and EnvelopeHessWork (envelope-Hessian accumulators).
# -------------------------------------------------------------------------------------------------

struct DemandWork
    # Forward-pass buffers
    U_blocks::Vector{Matrix{Float32}}    # CCPs (or utility pre-softmax) per block, N_bi × Z
    E_blocks::Vector{Matrix{Float32}}    # exp(η₁ + η₂·dist) per block, N_bi × Z
    resid_blocks::Vector{Matrix{Float32}} # counts - M_z·CCP per block, N_bi × Z
    inside_sum::Matrix{Float32}          # Σ_{k∈bi} CCP[k,z], n_blocks × Z
    α::Matrix{Float32}                    # Σ_bi X_bi · resid_bi, (K+1) × Z
    G_π::Matrix{Float32}                  # per-zip score wrt π, N_NONδ × Z
    score_η1::Vector{Float32}            # per-zip η₁ score
    score_η2::Vector{Float32}            # per-zip η₂ score
    # Envelope-Hessian accumulators (filled by eval_envelope_hessian!)
    partial_H::Matrix{Float32}
    K_corr::Matrix{Float32}
    H_per_thread::Vector{Matrix{Float32}}
    K_per_thread::Vector{Matrix{Float32}}
    singular_block::Vector{Bool}
end

function DemandWork(model::DemandModel)
    n_blocks = model.n_blocks
    Z = model.Z
    Dn = model.Dn
    Kp1 = model.K + 1
    N_NONδ = Kp1 * Dn + 2  # Πᵦ (Dn × K) + Πp (Dn) + η (2). Note: Kp1*Dn = K*Dn+Dn covers Πᵦ+Πp.
    nt = Threads.nthreads()
    return DemandWork(
        [zeros(Float32, length(model.block_param_idxs[bi]), Z) for bi in 1:n_blocks],
        [zeros(Float32, length(model.block_param_idxs[bi]), Z) for bi in 1:n_blocks],
        [zeros(Float32, length(model.block_param_idxs[bi]), Z) for bi in 1:n_blocks],
        zeros(Float32, n_blocks, Z),
        zeros(Float32, Kp1, Z),
        zeros(Float32, N_NONδ, Z),
        zeros(Float32, Z),
        zeros(Float32, Z),
        zeros(Float32, N_NONδ, N_NONδ),
        zeros(Float32, N_NONδ, N_NONδ),
        [zeros(Float32, N_NONδ, N_NONδ) for _ in 1:nt],
        [zeros(Float32, N_NONδ, N_NONδ) for _ in 1:nt],
        falses(n_blocks),
    )
end

# -------------------------------------------------------------------------------------------------
# Forward pass: utility → CCP per block.
# -------------------------------------------------------------------------------------------------

# Per-column softmax over a block's rows with outside-good utility = 0 (stabilized).
# Writes inside_sum[z] = Σ_k CCP[k,z].
@inline function softmax_block!(
    U::AbstractMatrix{Float32}, inside_sum_row::AbstractVector{Float32}
)
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

# Build Π'·D matrix (K+1 × Z) where Π = [Πᵦ | Πp]. Top K rows = Πᵦ'·D; last row = Πp'·D.
function compute_ΠtD!(
    ΠtD::Matrix{Float32}, Πᵦ::AbstractMatrix{Float32}, Πp::AbstractVector{Float32},
    D::Matrix{Float32},
)
    _, K = size(Πᵦ)
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
forward_pass!(work, model, params)
forward_pass!(work, model, δ, Πᵦ, Πp, η)

Computes per-block CCPs into `work.U_blocks` and exp distance utilities into `work.E_blocks`,
along with `work.inside_sum[bi, z] = Σ_{k∈block bi} CCP[k, z]`.

The 6-arg form is used by the counterfactual layer to extend `δ` past `model.N` (after
ModelAdd) without shuffling the non-δ block of `params`.

Multi-threaded across time blocks.
"""
function forward_pass!(work::DemandWork, model::DemandModel, params::AbstractVector{Float32})
    idx = param_indices(model)
    δ = view(params, idx.i_δ)
    Πᵦ = reshape(view(params, idx.i_Πᵦ), model.Dn, model.K)
    Πp = view(params, idx.i_Πp)
    η = view(params, idx.i_η)
    return forward_pass!(work, model, δ, Πᵦ, Πp, η)
end

function forward_pass!(
    work::DemandWork, model::DemandModel,
    δ::AbstractVector{Float32},
    Πᵦ::AbstractMatrix{Float32},
    Πp::AbstractVector{Float32},
    η::AbstractVector{Float32};
    blocks = 1:(model.n_blocks),
)
    η1, η2 = η[1], η[2]
    ΠtD = Matrix{Float32}(undef, model.K + 1, model.Z)
    compute_ΠtD!(ΠtD, Πᵦ, Πp, model.D_dense)

    @threads :static for bi in blocks
        U_bi = work.U_blocks[bi]
        E_bi = work.E_blocks[bi]
        X_bi = model.X_blocks[bi]
        dist_bi = model.dist_blocks[bi]
        param_idxs = model.block_param_idxs[bi]
        N_bi = size(U_bi, 1)

        mul!(U_bi, X_bi', ΠtD)
        @inbounds for z in 1:(model.Z)
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

# Inside-good ll for one block: Σ_{(k,z) ∈ nnz(counts)} counts[k,z] * log(max(CCP[k,z], ε)).
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
function _outside_ll(work::DemandWork, model::DemandModel)
    ll = 0.0f0
    @inbounds for z in 1:(model.Z)
        for bi in 1:(model.n_blocks)
            n_out = model.M_zs[z] - Float32(model.t_obs[bi, z])
            if n_out > 0.0f0
                p_out = max(1.0f0 - work.inside_sum[bi, z], 1.0f-10)
                ll += n_out * log(p_out)
            end
        end
    end
    return ll
end

"""
eval_ll!(work, model, params) → ll::Float32

Average log-likelihood per consumer-decision (sum × model.inv_N). Cheapest evaluator —
no resid, no gradient, no BHHH. Used by the inner-Newton line search.
"""
function eval_ll!(work::DemandWork, model::DemandModel, params::AbstractVector{Float32})
    forward_pass!(work, model, params)
    ll_in = 0.0f0
    for bi in 1:(model.n_blocks)
        ll_in += _inside_ll_block(work.U_blocks[bi], model.counts_blocks[bi])
    end
    return (ll_in + _outside_ll(work, model)) * model.inv_N
end

"""
eval_ll_δgrad!(work, model, params; δ_grad) → ll::Float32

Forward + average ll + δ-gradient (scaled by model.inv_N). Skips Π/η gradient and BHHH.
"""
function eval_ll_δgrad!(
    work::DemandWork, model::DemandModel, params::AbstractVector{Float32};
    δ_grad::AbstractVector{Float32},
)
    forward_pass!(work, model, params)
    fill!(δ_grad, 0.0f0)
    ll_in = 0.0f0
    @inbounds for bi in 1:(model.n_blocks)
        U_bi = work.U_blocks[bi]
        resid_bi = work.resid_blocks[bi]
        counts_bi = model.counts_blocks[bi]
        param_idxs = model.block_param_idxs[bi]
        N_bi = size(U_bi, 1)
        for z in 1:(model.Z)
            mz = model.M_zs[z]
            for k in 1:N_bi
                resid_bi[k, z] = -mz * U_bi[k, z]
            end
        end
        ll_block = 0.0f0
        for col in 1:(model.Z)
            for nz in counts_bi.colptr[col]:(counts_bi.colptr[col + 1] - 1)
                row = counts_bi.rowval[nz]
                val = Float32(counts_bi.nzval[nz])
                resid_bi[row, col] += val
                ll_block += val * log(max(U_bi[row, col], 1.0f-30))
            end
        end
        ll_in += ll_block
        for k in 1:N_bi
            s = 0.0f0
            for z in 1:(model.Z)
                s += resid_bi[k, z]
            end
            δ_grad[param_idxs[k]] = s
        end
    end
    ll = (ll_in + _outside_ll(work, model)) * model.inv_N
    δ_grad .*= model.inv_N
    return ll
end

# -------------------------------------------------------------------------------------------------
# Full combined evaluation: ll + gradient (all params) + BHHH (non-δ).
# -------------------------------------------------------------------------------------------------

"""
eval_combined!(work, model, params; grad, bhhh) → ll::Float32

Single pass: forward, ll, full gradient (∂ℓ/∂params), BHHH (∂ℓ/∂π outer products).
All return values are scaled by `model.inv_N` (avg likelihood, avg gradient, avg BHHH).
`work.U_blocks` still holds CCPs on exit (needed by Woodbury Newton step).
"""
function eval_combined!(
    work::DemandWork, model::DemandModel, params::AbstractVector{Float32};
    grad::AbstractVector{Float32}, bhhh::AbstractMatrix{Float32},
)
    forward_pass!(work, model, params)

    idx = param_indices(model)
    fill!(grad, 0.0f0)
    fill!(work.α, 0.0f0)
    fill!(work.score_η1, 0.0f0)
    fill!(work.score_η2, 0.0f0)

    ll_inside = 0.0f0
    for bi in 1:(model.n_blocks)
        U_bi = work.U_blocks[bi]
        E_bi = work.E_blocks[bi]
        resid_bi = work.resid_blocks[bi]
        X_bi = model.X_blocks[bi]
        dist_bi = model.dist_blocks[bi]
        counts_bi = model.counts_blocks[bi]
        param_idxs = model.block_param_idxs[bi]
        N_bi = size(U_bi, 1)

        @inbounds for z in 1:(model.Z)
            mz = model.M_zs[z]
            for k in 1:N_bi
                resid_bi[k, z] = -mz * U_bi[k, z]
            end
        end
        ll_block = 0.0f0
        @inbounds for col in 1:(model.Z)
            for nz in counts_bi.colptr[col]:(counts_bi.colptr[col + 1] - 1)
                row = counts_bi.rowval[nz]
                val = Float32(counts_bi.nzval[nz])
                resid_bi[row, col] += val
                ll_block += val * log(max(U_bi[row, col], 1.0f-30))
            end
        end
        ll_inside += ll_block

        @inbounds for k in 1:N_bi
            s = 0.0f0
            for z in 1:(model.Z)
                s += resid_bi[k, z]
            end
            grad[param_idxs[k]] = s
        end

        BLAS.gemm!('N', 'N', 1.0f0, X_bi, resid_bi, 1.0f0, work.α)

        @inbounds for z in 1:(model.Z)
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

    ll = ll_inside + _outside_ll(work, model)

    Π_grad = Matrix{Float32}(undef, model.Dn, model.K + 1)
    mul!(Π_grad, model.D_dense, work.α')

    i_Πᵦ_start = first(idx.i_Πᵦ)
    @inbounds for p in 1:(model.K), d in 1:(model.Dn)
        grad[i_Πᵦ_start + (p - 1) * model.Dn + (d - 1)] = Π_grad[d, p]
    end
    i_Πp_start = first(idx.i_Πp)
    @inbounds for d in 1:(model.Dn)
        grad[i_Πp_start + (d - 1)] = Π_grad[d, model.K + 1]
    end

    i_η_start = first(idx.i_η)
    grad[i_η_start] = sum(work.score_η1)
    grad[i_η_start + 1] = sum(work.score_η2)

    N_NONδ = size(work.G_π, 1)
    @inbounds for p in 1:(model.K), d in 1:(model.Dn)
        row = (p - 1) * model.Dn + d
        for z in 1:(model.Z)
            work.G_π[row, z] = model.D_dense[d, z] * work.α[p, z]
        end
    end
    @inbounds for d in 1:(model.Dn)
        row = model.K * model.Dn + d
        for z in 1:(model.Z)
            work.G_π[row, z] = model.D_dense[d, z] * work.α[model.K + 1, z]
        end
    end
    @inbounds for z in 1:(model.Z)
        work.G_π[N_NONδ - 1, z] = work.score_η1[z]
        work.G_π[N_NONδ, z] = work.score_η2[z]
    end

    fill!(bhhh, 0.0f0)
    BLAS.syrk!('U', 'N', 1.0f0, work.G_π, 0.0f0, bhhh)
    @inbounds for j in 1:N_NONδ, i in (j + 1):N_NONδ
        bhhh[i, j] = bhhh[j, i]
    end

    grad .*= model.inv_N
    bhhh .*= model.inv_N
    return ll * model.inv_N
end

# -------------------------------------------------------------------------------------------------
# Woodbury Newton step over δ. Block-diagonal Hessian (one block per time period).
# -------------------------------------------------------------------------------------------------

# Returns (-H_t)^{-1} g_t — Newton ascent direction. Falls back to g_t on singular middle matrix.
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
woodbury_newton_step!(step, work, model, g_δ)

Writes Newton ascent direction over δ (length ≥ N) into `step`, using `work.U_blocks`
(CCPs from latest forward pass) and per-block market sizes. `g_δ` is the *average*
δ-gradient (∂ℓ_avg/∂δ); the Newton direction is invariant to ll scaling, so we rescale
internally to undo the averaging.
"""
function woodbury_newton_step!(
    step::AbstractVector{Float32}, work::DemandWork, model::DemandModel,
    g_δ::AbstractVector{Float32},
)
    g_scale = 1.0f0 / model.inv_N
    @threads :static for bi in 1:(model.n_blocks)
        param_idxs = model.block_param_idxs[bi]
        g_t = Float32[g_δ[k] * g_scale for k in param_idxs]
        d_t = _woodbury_block(work.U_blocks[bi], model.M_zs, g_t)
        @inbounds for (k_local, p) in enumerate(param_idxs)
            step[p] = d_t[k_local]
        end
    end
    return step
end

# Multi-RHS Woodbury solve: W = (-H_t)⁻¹ · G; G is N_bi × n_rhs. Returns `true` on singular.
function _woodbury_solve_multi!(
    W::Matrix{Float32}, U::Matrix{Float32}, M_z::Vector{Float32}, G::Matrix{Float32}
)
    Z = length(M_z)
    N_bi = size(U, 1)
    @assert size(G, 1) == N_bi
    @assert size(W) == size(G)
    d = U * M_z
    inv_d = 1.0f0 ./ max.(d, 1.0f-12)
    sqrt_inv_d = sqrt.(inv_d)
    A = sqrt_inv_d .* U
    middle = zeros(Float32, Z, Z)
    middle[diagind(middle)] .= 1.0f0 ./ M_z
    BLAS.syrk!('U', 'T', -1.0f0, A, 1.0f0, middle)
    try
        F = cholesky!(Symmetric(middle, :U))
        SG = sqrt_inv_d .* G
        rhs = A' * SG
        w = F \ rhs
        W .= inv_d .* G .+ sqrt_inv_d .* (A * w)
        return false
    catch e
        e isa Union{LinearAlgebra.PosDefException, SingularException} || rethrow(e)
        W .= G
        return true
    end
end

# -------------------------------------------------------------------------------------------------
# Envelope-corrected concentrated Hessian
#   H̃_NLL = ∂²NLL/∂π² − C̃ · (∂²NLL/∂δ²)⁻¹ · C̃ᵀ
# Must run *after* a successful eval_combined! so U/E/resid buffers reflect current π.
# -------------------------------------------------------------------------------------------------

"""
eval_envelope_hessian!(work, model; H) → singular_blocks_count

Computes H̃_NLL into `H` (N_NONδ × N_NONδ). Returns count of time blocks that hit a singular
Woodbury solve (W=G fallback — degrades gracefully toward BHHH-like).
"""
function eval_envelope_hessian!(
    work::DemandWork, model::DemandModel; H::AbstractMatrix{Float32}
)
    N_NONδ = (model.K + 1) * model.Dn + 2
    @assert size(H, 1) == size(H, 2) == N_NONδ
    Kx = model.K
    Dn = model.Dn
    Z = model.Z
    row_η1 = N_NONδ - 1
    row_η2 = N_NONδ

    for buf in work.H_per_thread
        fill!(buf, 0.0f0)
    end
    for buf in work.K_per_thread
        fill!(buf, 0.0f0)
    end
    fill!(work.singular_block, false)

    @threads :static for bi in 1:(model.n_blocks)
        tid = Threads.threadid()
        H_local = work.H_per_thread[tid % Threads.nthreads() + 1]
        K_local = work.K_per_thread[tid % Threads.nthreads() + 1]

        U_bi = work.U_blocks[bi]
        E_bi = work.E_blocks[bi]
        resid_bi = work.resid_blocks[bi]
        X_bi = model.X_blocks[bi]
        dist_bi = model.dist_blocks[bi]
        N_bi = size(U_bi, 1)
        Kp1 = Kx + 1

        xbar = X_bi * U_bi
        XEbar = zeros(Float32, Kp1, Z)
        XdEbar = zeros(Float32, Kp1, Z)
        Ebar = zeros(Float32, Z)
        dEbar = zeros(Float32, Z)
        EEbar = zeros(Float32, Z)
        dEEbar = zeros(Float32, Z)
        dEdEbar = zeros(Float32, Z)

        PX = Matrix{Float32}(undef, Kp1, N_bi)
        Mzx = Matrix{Float32}(undef, Kp1, Kp1)
        Vz = Matrix{Float32}(undef, Kp1, Kp1)

        @inbounds for z in 1:Z
            Mz = model.M_zs[z]
            for j in 1:N_bi
                Pjz = U_bi[j, z]
                Ejz = E_bi[j, z]
                Djz = dist_bi[j, z]
                for k in 1:Kp1
                    PX[k, j] = Pjz * X_bi[k, j]
                end
                Ebar[z] += Pjz * Ejz
                dEbar[z] += Pjz * Djz * Ejz
                EEbar[z] += Pjz * Ejz * Ejz
                dEEbar[z] += Pjz * Djz * Ejz * Ejz
                dEdEbar[z] += Pjz * Djz * Djz * Ejz * Ejz
                for k in 1:Kp1
                    XEbar[k, z] += Pjz * X_bi[k, j] * Ejz
                    XdEbar[k, z] += Pjz * X_bi[k, j] * Djz * Ejz
                end
            end

            BLAS.gemm!('N', 'T', 1.0f0, PX, X_bi, 0.0f0, Mzx)
            for k2 in 1:Kp1, k1 in 1:Kp1
                Vz[k1, k2] = Mzx[k1, k2] - xbar[k1, z] * xbar[k2, z]
            end
            for k2 in 1:Kp1, k1 in 1:Kp1
                coef = Mz * Vz[k1, k2]
                col_off = (k2 - 1) * Dn
                row_off = (k1 - 1) * Dn
                for d2 in 1:Dn
                    dd = coef * model.D_dense[d2, z]
                    for d1 in 1:Dn
                        H_local[row_off + d1, col_off + d2] += dd * model.D_dense[d1, z]
                    end
                end
            end
            H_local[row_η1, row_η1] += Mz * (EEbar[z] - Ebar[z]^2)
            mix = Mz * (dEEbar[z] - Ebar[z] * dEbar[z])
            H_local[row_η1, row_η2] += mix
            H_local[row_η2, row_η1] += mix
            H_local[row_η2, row_η2] += Mz * (dEdEbar[z] - dEbar[z]^2)
        end

        diff_E = XEbar .- xbar .* reshape(Ebar, 1, Z)
        diff_dE = XdEbar .- xbar .* reshape(dEbar, 1, Z)
        Mz_row = reshape(model.M_zs, 1, Z)
        M_diff_E = diff_E .* Mz_row
        M_diff_dE = diff_dE .* Mz_row
        D_diff_E = model.D_dense * M_diff_E'
        D_diff_dE = model.D_dense * M_diff_dE'
        @inbounds for k in 1:Kp1, d in 1:Dn
            row_idx = (k - 1) * Dn + d
            H_local[row_idx, row_η1] -= D_diff_E[d, k]
            H_local[row_η1, row_idx] -= D_diff_E[d, k]
            H_local[row_idx, row_η2] -= D_diff_dE[d, k]
            H_local[row_η2, row_idx] -= D_diff_dE[d, k]
        end

        s_rE = 0.0f0
        s_rdE = 0.0f0
        s_rddE = 0.0f0
        @inbounds for z in 1:Z, j in 1:N_bi
            r = resid_bi[j, z]
            e = E_bi[j, z]
            di = dist_bi[j, z]
            s_rE += r * e
            s_rdE += r * di * e
            s_rddE += r * di * di * e
        end
        H_local[row_η1, row_η1] += s_rE
        H_local[row_η1, row_η2] += s_rdE
        H_local[row_η2, row_η1] += s_rdE
        H_local[row_η2, row_η2] += s_rddE

        Ctilde = zeros(Float32, N_NONδ, N_bi)
        MzU = model.M_zs .* U_bi'
        S1 = model.D_dense * MzU
        @inbounds for k in 1:Kp1
            Mxk = model.M_zs .* view(xbar, k, :)
            T_k = model.D_dense .* reshape(Mxk, 1, Z)
            S2_k = T_k * U_bi'
            for d in 1:Dn, j in 1:N_bi
                row_idx = (k - 1) * Dn + d
                Ctilde[row_idx, j] = X_bi[k, j] * S1[d, j] - S2_k[d, j]
            end
        end

        MzEbar = model.M_zs .* Ebar
        MzdEbar = model.M_zs .* dEbar
        term1_η1 = U_bi * MzEbar
        term1_η2 = U_bi * MzdEbar
        @inbounds for j in 1:N_bi
            s_e = 0.0f0
            s_de = 0.0f0
            for z in 1:Z
                Mz = model.M_zs[z]
                Pjz = U_bi[j, z]
                Ejz = E_bi[j, z]
                Djz = dist_bi[j, z]
                s_e += Mz * Pjz * Ejz
                s_de += Mz * Pjz * Djz * Ejz
            end
            Ctilde[row_η1, j] = term1_η1[j] - s_e
            Ctilde[row_η2, j] = term1_η2[j] - s_de
        end

        Gt = Matrix(Ctilde')
        W = Matrix{Float32}(undef, N_bi, N_NONδ)
        singular = _woodbury_solve_multi!(W, U_bi, model.M_zs, Gt)
        work.singular_block[bi] = singular
        BLAS.gemm!('N', 'N', 1.0f0, Ctilde, W, 1.0f0, K_local)
    end

    fill!(work.partial_H, 0.0f0)
    fill!(work.K_corr, 0.0f0)
    for buf in work.H_per_thread
        work.partial_H .+= buf
    end
    for buf in work.K_per_thread
        work.K_corr .+= buf
    end

    @inbounds for j in 1:N_NONδ, i in 1:N_NONδ
        H[i, j] = (work.partial_H[i, j] - work.K_corr[i, j]) * model.inv_N
    end
    @inbounds for j in 1:N_NONδ, i in (j + 1):N_NONδ
        s = 0.5f0 * (H[i, j] + H[j, i])
        H[i, j] = s
        H[j, i] = s
    end
    return count(work.singular_block)
end

# Eigenvalue floor for the envelope-corrected Hessian.
function definiteness_shift!(H::AbstractMatrix{Float32}, floor::Float32 = 1.0f-3)
    N = size(H, 1)
    Hd = Symmetric(Float64.(H))
    λmin = Float32(eigmin(Hd))
    if λmin < floor
        shift = floor - λmin
        @inbounds for i in 1:N
            H[i, i] += shift
        end
        return shift
    end
    return 0.0f0
end

# -------------------------------------------------------------------------------------------------
# Tweak system: PriceMultiply / ModelAdd / ModelDelete
# -------------------------------------------------------------------------------------------------

abstract type AbstractTweak end

@kwdef struct PriceMultiply <: AbstractTweak
    dealer_uid::Union{UUID, Nothing} = nothing
    vehicle::Union{VehicleKey, Nothing} = nothing
    quarter::Union{QuarterKey, Nothing} = nothing
    by::Float32
end

@kwdef struct PriceChange <: AbstractTweak
    dealer_uid::Union{UUID, Nothing} = nothing
    vehicle::Union{VehicleKey, Nothing} = nothing
    quarter::Union{QuarterKey, Nothing} = nothing
    price::Float32
end

@kwdef struct ModelAdd <: AbstractTweak
    vehicle::VehicleKey
    quarter::QuarterKey
    dealer_uid::UUID
    price::Union{Float32, Nothing} = nothing
end

@kwdef struct ModelDelete <: AbstractTweak
    vehicle::VehicleKey
    quarter::QuarterKey
    dealer_uid::UUID
end

macro getfe(model, what)
    what_sym = string(what)
    what_fe_sym = "fe_" * string(what)
    quote
        pluck(unique($(model).fe, [$what_sym, $what_fe_sym]), $what_sym => ByRow(==($what)))[$what_fe_sym]
    end
end

# FixedEffectModels doesn't handle OOS predictions, even though they're possible
function pred_regout(model, df)
    out = modelmatrix(model.formula_schema, df) * model.coef
    for (i, (; year, quarter, vehicleid, dealer_uid)) in enumerate(eachrow(df))
        out[i] += @getfe model year
        out[i] += @getfe model quarter
        out[i] += @getfe model vehicleid
        out[i] += @getfe model dealer_uid
    end
    return out
end

# Helper: look up flat δ indices matching a tweak's filters. Filters out products that
# have been deleted (not in tjf_to_block_k). Returns `missing` if no match.
function resolve_indices(model::DemandModel, tweak::AbstractTweak)
    # Fast path: all three filters set ⇒ at most one product. Direct tjf lookup avoids
    # intersecting three potentially-large index vectors (hot path inside residual_demand).
    if !isnothing(tweak.dealer_uid) && !isnothing(tweak.vehicle) && !isnothing(tweak.quarter)
        tjf_key = (tweak.quarter, tweak.vehicle, tweak.dealer_uid)
        idx = get(model.product_lookup.tjf, tjf_key, Int32(0))
        idx == Int32(0) && return missing
        haskey(model.tjf_to_block_k, idx) || return missing
        return [model.tjf_to_block_k[idx]], Int32[idx]
    end
    indices = Vector{Int32}[]
    if !isnothing(tweak.dealer_uid)
        push!(indices, get(model.product_lookup.f, tweak.dealer_uid, Int32[]))
    end
    if !isnothing(tweak.vehicle)
        push!(indices, get(model.product_lookup.j, tweak.vehicle, Int32[]))
    end
    if !isnothing(tweak.quarter)
        push!(indices, get(model.product_lookup.t, tweak.quarter, Int32[]))
    end
    isempty(indices) && return missing
    δ_idxs_all = reduce(intersect, indices)
    δ_idxs = Int32[i for i in δ_idxs_all if haskey(model.tjf_to_block_k, i)]
    isempty(δ_idxs) && return missing
    dense_indices = [model.tjf_to_block_k[i] for i in δ_idxs]
    return dense_indices, δ_idxs
end

function _block_index_for_quarter(model::DemandModel, quarter::QuarterKey)
    bi = findfirst(==(quarter), model.block_keys)
    isnothing(bi) && error("Quarter $quarter not present in model time blocks")
    return Int(bi)
end

# Sparse-row primitives for counts_blocks[bi] (N_bi × Z).
function _extract_sparse_row(M::SparseMatrixCSC{Tv, Ti}, k::Integer) where {Tv, Ti}
    pairs = Tuple{Ti, Tv}[]
    @inbounds for col in 1:size(M, 2)
        for nz in M.colptr[col]:(M.colptr[col + 1] - 1)
            M.rowval[nz] == Ti(k) && push!(pairs, (Ti(col), M.nzval[nz]))
        end
    end
    return pairs
end

function _drop_sparse_row(M::SparseMatrixCSC{Tv, Ti}, k::Integer) where {Tv, Ti}
    I, J, V = findnz(M)
    mask = I .!= Ti(k)
    I_new = I[mask]
    @inbounds for i in eachindex(I_new)
        I_new[i] > Ti(k) && (I_new[i] -= one(Ti))
    end
    return sparse(I_new, J[mask], V[mask], size(M, 1) - 1, size(M, 2))
end

function _insert_sparse_row(
    M::SparseMatrixCSC{Tv, Ti}, k::Integer, row_pairs::Vector{Tuple{Ti, Tv}}
) where {Tv, Ti}
    I, J, V = findnz(M)
    @inbounds for i in eachindex(I)
        I[i] >= Ti(k) && (I[i] += one(Ti))
    end
    for (col, val) in row_pairs
        push!(I, Ti(k))
        push!(J, col)
        push!(V, val)
    end
    return sparse(I, J, V, size(M, 1) + 1, size(M, 2))
end

function _append_empty_sparse_row(M::SparseMatrixCSC{Tv, Ti}) where {Tv, Ti}
    return SparseMatrixCSC{Tv, Ti}(
        size(M, 1) + 1, size(M, 2), copy(M.colptr), copy(M.rowval), copy(M.nzval)
    )
end

function _pop_last_sparse_row(M::SparseMatrixCSC{Tv, Ti}) where {Tv, Ti}
    last_row = Ti(size(M, 1))
    @inbounds for r in M.rowval
        r == last_row && error("_pop_last_sparse_row: last row has nonzero entries")
    end
    return SparseMatrixCSC{Tv, Ti}(
        size(M, 1) - 1, size(M, 2), copy(M.colptr), copy(M.rowval), copy(M.nzval)
    )
end

function _resize_block_buffers!(
    work::DemandWork, model::DemandModel, bi::Integer, N_bi_new::Integer
)
    work.U_blocks[bi] = zeros(Float32, N_bi_new, model.Z)
    work.E_blocks[bi] = zeros(Float32, N_bi_new, model.Z)
    work.resid_blocks[bi] = zeros(Float32, N_bi_new, model.Z)
    return nothing
end

# Majority-vote body class for a (make, model, model_year) using existing product_info entries.
function _body_flags_for_vehicle(model::DemandModel, vehicle::VehicleKey)
    j_base = vehicle[1:3]
    counts = (coupe = 0, suv = 0, pickup = 0, sedan = 0)
    found = false
    @inbounds for (key, idx) in pairs(model.product_lookup.tjf)
        (_, j_k, _) = key
        j_k[1:3] == j_base || continue
        haskey(model.tjf_to_block_k, idx) || continue
        info = model.product_info[idx]
        counts = (
            coupe = counts.coupe + Int(info.is_coupe),
            suv = counts.suv + Int(info.is_suv),
            pickup = counts.pickup + Int(info.is_pickup),
            sedan = counts.sedan + Int(info.is_sedan),
        )
        found = true
    end
    total = counts.coupe + counts.suv + counts.pickup + counts.sedan
    if !found || total == 0
        @warn "No body-class data for $(j_base); defaulting to sedan"
        return (false, false, false, true)
    end
    maxc = max(counts.coupe, counts.suv, counts.pickup, counts.sedan)
    is_coupe = counts.coupe == maxc
    is_suv = !is_coupe && counts.suv == maxc
    is_pickup = !is_coupe && !is_suv && counts.pickup == maxc
    is_sedan = !is_coupe && !is_suv && !is_pickup
    return (is_coupe, is_suv, is_pickup, is_sedan)
end

# ---- apply!(::PriceMultiply, ...) ----

"""
apply!(tweak::PriceMultiply, model, δ; β_price) → undo handle (or `missing` if no match)

Multiplies the standardized price in `model.X_blocks[bi][end, k]` by `tweak.by` for every
(t, j, f) matching the tweak's filters, and shifts `δ[δ_idx]` by `β_price · old_price · (by − 1)`.
"""
function apply!(
    tweak::PriceMultiply, model::DemandModel, δ::AbstractVector{Float32}; β_price::Float32
)
    resolved = resolve_indices(model, tweak)
    ismissing(resolved) && return missing
    dense_indices, δ_idxs = resolved
    K = model.K
    by = tweak.by
    n = length(dense_indices)
    δ_offsets = Vector{Float32}(undef, n)
    old_prices = Vector{Float32}(undef, n)
    @inbounds for (i, (bi, k_local)) in enumerate(dense_indices)
        old_p = model.X_blocks[bi][K + 1, k_local]
        old_prices[i] = old_p
        model.X_blocks[bi][K + 1, k_local] = old_p * by
        offset = β_price * old_p * (by - 1.0f0)
        δ_offsets[i] = offset
        δ[δ_idxs[i]] += offset
    end
    return (;
        kind = :price_multiply,
        dense_indices = collect(dense_indices),
        δ_indices = collect(δ_idxs),
        δ_offsets,
        old_prices,
        by,
    )
end

# ---- apply!(::PriceChange, ...) ----

"""
apply!(tweak::PriceChange, model, δ; β_price) → undo handle (or `missing` if no match)

Wholesale-sets the standardized price in `model.X_blocks[bi][end, k]` to `tweak.price` for
every (t, j, f) matching the tweak's filters, and shifts `δ[δ_idx]` by
`β_price · (tweak.price − old_price)`.
"""
function apply!(
    tweak::PriceChange, model::DemandModel, δ::AbstractVector{Float32}; β_price::Float32
)
    resolved = resolve_indices(model, tweak)
    ismissing(resolved) && return missing
    dense_indices, δ_idxs = resolved
    K = model.K
    new_p = tweak.price
    n = length(dense_indices)
    δ_offsets = Vector{Float32}(undef, n)
    old_prices = Vector{Float32}(undef, n)
    @inbounds for (i, (bi, k_local)) in enumerate(dense_indices)
        old_p = model.X_blocks[bi][K + 1, k_local]
        old_prices[i] = old_p
        model.X_blocks[bi][K + 1, k_local] = new_p
        offset = β_price * (new_p - old_p)
        δ_offsets[i] = offset
        δ[δ_idxs[i]] += offset
    end
    return (;
        kind = :price_change,
        dense_indices = collect(dense_indices),
        δ_indices = collect(δ_idxs),
        δ_offsets,
        old_prices,
        new_price = new_p,
    )
end

# ---- apply!(::ModelDelete, ...) ----

"""
apply!(tweak::ModelDelete, model, work, δ) → undo handle (or `missing` if not found)

Removes the (t, j, f) product from its time block, drops its X column / dist row /
counts row, decrements t_obs by the dropped counts, and removes its entry from
block_param_idxs, tjf_to_block_k, and product_lookup.{t, j, f, make}. The δ entry is
left orphaned in `δ` (use the 6-arg forward_pass!).
"""
function apply!(
    tweak::ModelDelete, model::DemandModel, work::DemandWork, δ::AbstractVector{Float32}
)
    tjf_key = (tweak.quarter, tweak.vehicle, tweak.dealer_uid)
    δ_idx_raw = get(model.product_lookup.tjf, tjf_key, Int32(0))
    if δ_idx_raw == 0 || !haskey(model.tjf_to_block_k, δ_idx_raw)
        @warn "ModelDelete: (t, j, f) $tjf_key not found; no-op"
        return missing
    end
    δ_idx = Int(δ_idx_raw)
    bi_t, k_local_t = model.tjf_to_block_k[δ_idx_raw]
    bi = Int(bi_t)
    k_local = Int(k_local_t)
    Z = model.Z
    N_bi = length(model.block_param_idxs[bi])

    X_col = model.X_blocks[bi][:, k_local] |> copy
    dist_col = model.dist_blocks[bi][k_local, :] |> copy
    counts_col_pairs = _extract_sparse_row(model.counts_blocks[bi], k_local)
    δ_value = δ[δ_idx]

    keep_cols = [c for c in 1:N_bi if c != k_local]
    model.X_blocks[bi] = model.X_blocks[bi][:, keep_cols]
    model.dist_blocks[bi] = model.dist_blocks[bi][keep_cols, :]
    @inbounds for (col_t, val_t) in counts_col_pairs
        model.t_obs[bi, Int(col_t)] -= Int32(val_t)
    end
    model.counts_blocks[bi] = _drop_sparse_row(model.counts_blocks[bi], k_local)
    deleteat!(model.block_param_idxs[bi], k_local)
    delete!(model.tjf_to_block_k, δ_idx_raw)
    for (key, (b_t, k_t)) in pairs(model.tjf_to_block_k)
        if Int(b_t) == bi && Int(k_t) > k_local
            model.tjf_to_block_k[key] = (b_t, k_t - Int32(1))
        end
    end

    # Remove δ_idx from product_lookup.{t, j, f, make} (record positions for undo)
    t_pos = findfirst(==(δ_idx_raw), model.product_lookup.t[tweak.quarter])
    j_pos = findfirst(==(δ_idx_raw), model.product_lookup.j[tweak.vehicle])
    f_pos = findfirst(==(δ_idx_raw), model.product_lookup.f[tweak.dealer_uid])
    make_key = first(tweak.vehicle)
    make_pos = findfirst(==(δ_idx_raw), model.product_lookup.make[make_key])
    deleteat!(model.product_lookup.t[tweak.quarter], t_pos)
    deleteat!(model.product_lookup.j[tweak.vehicle], j_pos)
    deleteat!(model.product_lookup.f[tweak.dealer_uid], f_pos)
    deleteat!(model.product_lookup.make[make_key], make_pos)

    _resize_block_buffers!(work, model, bi, N_bi - 1)
    return (;
        kind = :model_delete,
        bi,
        k_local,
        δ_idx = Int32(δ_idx),
        δ_value,
        X_col,
        dist_col,
        counts_col_pairs,
        tjf_key,
        t_pos, j_pos, f_pos, make_pos,
    )
end

# ---- apply!(::ModelAdd, ...) ----

"""
apply!(tweak::ModelAdd, model, work, δ; regout, first_stage, J_t_lookup, hausman_fn) → undo handle

Predicts Hausman instrument via `hausman_fn`, predicts price via `first_stage`, predicts δ via
`regout`. Appends a new column to X_blocks[bi], a new row to dist_blocks[bi], an empty row to
counts_blocks[bi]; pushes the predicted δ onto `δ`; registers in block_param_idxs[bi],
tjf_to_block_k, and product_lookup.{t, j, f, make, tjf}. Returns `missing` on prediction failure.
"""
function apply!(
    tweak::ModelAdd, model::DemandModel, work::DemandWork, δ::Vector{Float32};
    regout, first_stage, J_t_lookup::AbstractDict{Tuple{Int64, Int64}, <:Integer},
    hausman_fn::Function,
)
    tjf_key = (tweak.quarter, tweak.vehicle, tweak.dealer_uid)
    existing = get(model.product_lookup.tjf, tjf_key, Int32(0))
    if existing != 0 && haskey(model.tjf_to_block_k, existing)
        @warn "ModelAdd: (t, j, f) $tjf_key already present; no-op"
        return missing
    end
    if !(tweak.dealer_uid in names(model.distances, 2))
        @warn "ModelAdd: dealer $(tweak.dealer_uid) not in distance matrix; no-op"
        return missing
    end

    if !haskey(J_t_lookup, tweak.quarter)
        @warn "ModelAdd: J_t_lookup missing quarter $(tweak.quarter); no-op"
        return missing
    end
    J_t = Int(J_t_lookup[tweak.quarter])
    year_val, quarter_val = tweak.quarter
    make_str, model_str, model_year, trim_rank = tweak.vehicle
    vehicleid = string(model_str, "_", model_year, "_", trim_rank)

    p_hat::Float32 = if isnothing(tweak.price)
        j_for_hausman = (make_str, model_str, model_year)
        k_for_hausman = Float64(trim_rank)
        p_hausman_raw = hausman_fn(tweak.quarter, j_for_hausman, k_for_hausman, tweak.dealer_uid)
        if isnan(p_hausman_raw)
            @warn "ModelAdd: hausman_fn returned NaN for $tjf_key; no-op"
            return missing
        end
        fs_row = DataFrame(;
            price_hausman = [Float64(p_hausman_raw)],
            J = [J_t],
            vehicleid = [vehicleid],
            dealer_uid = [tweak.dealer_uid],
            year = [year_val],
            quarter = [quarter_val],
        )
        p_hat_raw = predict(first_stage, fs_row)
        if ismissing(p_hat_raw) || ismissing(first(p_hat_raw))
            @warn "ModelAdd: first_stage predict returned missing for $tjf_key; no-op"
            return missing
        end
        Float32(first(p_hat_raw))
    else
        tweak.price
    end

    # regout's structural regressor is price; price_hausman is the IV. FixedEffectModels.predict
    # may read all formula RHS columns, so include both even when price was given directly.
    row_df = DataFrame(;
        price = [Float64(p_hat)],
        price_hausman = [Float64(p_hat)],
        J = [J_t],
        vehicleid = [vehicleid],
        dealer_uid = [tweak.dealer_uid],
        year = [year_val],
        quarter = [quarter_val],
    )
    δ_hat_raw = pred_regout(regout, row_df)
    if ismissing(δ_hat_raw) || ismissing(first(δ_hat_raw))
        @warn "ModelAdd: regout predict returned missing for $tjf_key; no-op"
        return missing
    end
    δ_hat = Float32(first(δ_hat_raw))

    bi = _block_index_for_quarter(model, tweak.quarter)
    K = model.K
    Z = model.Z
    N_bi = length(model.block_param_idxs[bi])
    is_coupe, is_suv, is_pickup, is_sedan = _body_flags_for_vehicle(model, tweak.vehicle)

    new_X_col = Vector{Float32}(undef, K + 1)
    body_flags = (is_coupe, is_suv, is_pickup, is_sedan)
    @inbounds for (p_i, attr) in enumerate(PRODUCT_ATTRIBUTES)
        new_X_col[p_i] = Float32(
            if attr === :is_coupe
                body_flags[1]
            elseif attr === :is_suv
                body_flags[2]
            elseif attr === :is_pickup
                body_flags[3]
            else
                body_flags[4]
            end,
        )
    end
    new_X_col[K + 1] = p_hat

    new_dist_row = Vector{Float32}(undef, Z)
    @inbounds for (zi, z) in enumerate(model.zips)
        new_dist_row[zi] = Float32(model.distances[z, tweak.dealer_uid])
    end

    model.X_blocks[bi] = hcat(model.X_blocks[bi], new_X_col)
    model.dist_blocks[bi] = vcat(model.dist_blocks[bi], reshape(new_dist_row, 1, Z))
    model.counts_blocks[bi] = _append_empty_sparse_row(model.counts_blocks[bi])

    push!(δ, δ_hat)
    new_δ_idx = Int32(length(δ))
    push!(model.block_param_idxs[bi], new_δ_idx)
    k_new_local = Int32(N_bi + 1)
    model.tjf_to_block_k[new_δ_idx] = (Int32(bi), k_new_local)

    # Register in product_lookup
    push!(get!(model.product_lookup.t, tweak.quarter, Int32[]), new_δ_idx)
    push!(get!(model.product_lookup.j, tweak.vehicle, Int32[]), new_δ_idx)
    push!(get!(model.product_lookup.f, tweak.dealer_uid, Int32[]), new_δ_idx)
    push!(get!(model.product_lookup.make, make_str, Int32[]), new_δ_idx)
    tjf_was_present = existing != 0
    prev_tjf_value = existing
    model.product_lookup.tjf[tjf_key] = new_δ_idx

    _resize_block_buffers!(work, model, bi, N_bi + 1)
    return (;
        kind = :model_add,
        bi,
        k_local = Int(k_new_local),
        δ_idx = new_δ_idx,
        tjf_key,
        tjf_was_present,
        prev_tjf_value,
    )
end

# ---- undo! ----

"""
undo!(handle, model, δ)
undo!(handle, model, work, δ)

Reverses an `apply!` call. The 3-arg form supports `:price_multiply`; the 4-arg form supports
`:model_delete` and `:model_add`.
"""
function undo!(handle::NamedTuple, model::DemandModel, δ::AbstractVector{Float32})
    handle.kind in (:price_multiply, :price_change) ||
        error(
            "undo!(handle, model, δ): only supports :price_multiply / :price_change; got $(handle.kind)"
        )
    K = model.K
    @inbounds for (i, (bi, k_local)) in enumerate(handle.dense_indices)
        model.X_blocks[bi][K + 1, k_local] = handle.old_prices[i]
        δ[handle.δ_indices[i]] -= handle.δ_offsets[i]
    end
    return nothing
end

function undo!(
    handle::NamedTuple, model::DemandModel, work::DemandWork, δ::Vector{Float32}
)
    if handle.kind == :model_delete
        bi = handle.bi
        k_local = handle.k_local
        N_bi_now = length(model.block_param_idxs[bi])
        Xb = model.X_blocks[bi]
        before = view(Xb, :, 1:(k_local - 1))
        after = view(Xb, :, k_local:N_bi_now)
        model.X_blocks[bi] = hcat(before, reshape(handle.X_col, :, 1), after)
        Db = model.dist_blocks[bi]
        before_r = view(Db, 1:(k_local - 1), :)
        after_r = view(Db, k_local:N_bi_now, :)
        model.dist_blocks[bi] = vcat(before_r, reshape(handle.dist_col, 1, :), after_r)
        model.counts_blocks[bi] = _insert_sparse_row(
            model.counts_blocks[bi], k_local, handle.counts_col_pairs
        )
        @inbounds for (col_t, val_t) in handle.counts_col_pairs
            model.t_obs[bi, Int(col_t)] += Int32(val_t)
        end
        for (key, (b_t, k_t)) in pairs(model.tjf_to_block_k)
            if Int(b_t) == bi && Int(k_t) >= k_local
                model.tjf_to_block_k[key] = (b_t, k_t + Int32(1))
            end
        end
        insert!(model.block_param_idxs[bi], k_local, handle.δ_idx)
        model.tjf_to_block_k[handle.δ_idx] = (Int32(bi), Int32(k_local))
        δ[handle.δ_idx] = handle.δ_value
        # Restore product_lookup entries
        tquarter, tvehicle, tdealer = handle.tjf_key
        make_key = first(tvehicle)
        insert!(model.product_lookup.t[tquarter], handle.t_pos, handle.δ_idx)
        insert!(model.product_lookup.j[tvehicle], handle.j_pos, handle.δ_idx)
        insert!(model.product_lookup.f[tdealer], handle.f_pos, handle.δ_idx)
        insert!(model.product_lookup.make[make_key], handle.make_pos, handle.δ_idx)
        _resize_block_buffers!(work, model, bi, N_bi_now + 1)
        return nothing
    elseif handle.kind == :model_add
        bi = handle.bi
        k_local = handle.k_local
        N_bi_now = length(model.block_param_idxs[bi])
        k_local == N_bi_now ||
            error("undo!(:model_add): block_param_idxs unexpectedly modified")
        model.X_blocks[bi] = model.X_blocks[bi][:, 1:(N_bi_now - 1)]
        model.dist_blocks[bi] = model.dist_blocks[bi][1:(N_bi_now - 1), :]
        model.counts_blocks[bi] = _pop_last_sparse_row(model.counts_blocks[bi])
        pop!(model.block_param_idxs[bi])
        delete!(model.tjf_to_block_k, handle.δ_idx)
        Int(handle.δ_idx) == length(δ) ||
            error("undo!(:model_add): δ has been modified since apply")
        pop!(δ)
        # Reverse product_lookup pushes
        tquarter, tvehicle, tdealer = handle.tjf_key
        make_key = first(tvehicle)
        pop!(model.product_lookup.t[tquarter])
        pop!(model.product_lookup.j[tvehicle])
        pop!(model.product_lookup.f[tdealer])
        pop!(model.product_lookup.make[make_key])
        if handle.tjf_was_present
            model.product_lookup.tjf[handle.tjf_key] = handle.prev_tjf_value
        else
            delete!(model.product_lookup.tjf, handle.tjf_key)
        end
        _resize_block_buffers!(work, model, bi, N_bi_now - 1)
        return nothing
    else
        error("undo!: unknown handle kind $(handle.kind)")
    end
end

# -------------------------------------------------------------------------------------------------
# Counterfactual outputs: expected_demand, share_matrix, foc_residual
# -------------------------------------------------------------------------------------------------

function dealer_block_locals(model::DemandModel, dealer_uid::UUID, t::QuarterKey)
    bi = _block_index_for_quarter(model, t)
    dealer_idxs = get(model.product_lookup.f, dealer_uid, Int32[])
    quarter_idxs = get(model.product_lookup.t, t, Int32[])
    matching = Int32[
        i for i in intersect(dealer_idxs, quarter_idxs) if haskey(model.tjf_to_block_k, i)
    ]
    isempty(matching) && return bi, Int[], Int32[]
    pairs_ = [(model.tjf_to_block_k[idx], idx) for idx in matching]
    sort!(pairs_; by = p -> Int(p[1][2]))
    k_locals = Int[Int(p[1][2]) for p in pairs_]
    δ_idxs = Int32[p[2] for p in pairs_]
    return bi, k_locals, δ_idxs
end

"""
share_matrix(model, work, params, dealer_uid, t) → Matrix{Float32}

`N_d × N_d` Jacobian of expected demand w.r.t. standardized prices for `dealer_uid` in quarter `t`.

  J[i, j] = ∂E[demand_i] / ∂price'_j
          = Σ_z M_z · α_z · P_i(z) · (1_{i=j} - P_j(z))
          where α_z = Σ_d Πp[d] · D_z[d].
"""
function share_matrix(
    model::DemandModel, work::DemandWork, params::AbstractVector{Float32},
    dealer_uid::UUID, t::QuarterKey; mean_α,
)
    bi, k_locals, _ = dealer_block_locals(model, dealer_uid, t)
    N_d = length(k_locals)
    N_d == 0 && return Matrix{Float32}(undef, 0, 0)

    idx = param_indices(model)
    Πp = view(params, idx.i_Πp)
    Z = model.Z
    α = Vector{Float32}(undef, Z)
    @inbounds for z in 1:Z
        s = 0.0f0
        for d in 1:(model.Dn)
            s += Πp[d] * model.D_dense[d, z]
        end
        α[z] = s + mean_α
    end

    P = work.U_blocks[bi]
    P_d = P[k_locals, :]
    w = model.M_zs .* α
    PdW = P_d .* reshape(w, 1, Z)
    J = -PdW * P_d'
    @inbounds for i in 1:N_d
        s_diag = 0.0f0
        for z in 1:Z
            s_diag += w[z] * P_d[i, z]
        end
        J[i, i] += s_diag
    end
    return J
end

"""
expected_demand(model, work, dealer_uid, t) → Vector{Float32}

q_d[i] = Σ_z M_z · P_d[i, z], same indexing as `share_matrix`.
"""
function expected_demand(
    model::DemandModel, work::DemandWork, dealer_uid::UUID, t::QuarterKey
)
    bi, k_locals, _ = dealer_block_locals(model, dealer_uid, t)
    N_d = length(k_locals)
    N_d == 0 && return Float32[]
    P = work.U_blocks[bi]
    q = zeros(Float32, N_d)
    Z = model.Z
    @inbounds for i in 1:N_d
        ki = k_locals[i]
        s = 0.0f0
        for z in 1:Z
            s += model.M_zs[z] * P[ki, z]
        end
        q[i] = s
    end
    return q
end

"""
foc_residual(model, work, params, dealer_uid, t, prices, costs) → Vector{Float32}

Bertrand-Nash FOC residual: `q_d + Jᵀ · (prices - costs)`. Zero at equilibrium. The caller is
responsible for applying any price tweaks and re-running `forward_pass!` first.
"""
function foc_residual(
    model::DemandModel, work::DemandWork, params::AbstractVector{Float32},
    dealer_uid::UUID, t::QuarterKey,
    prices::AbstractVector{Float32}, costs::AbstractVector{Float32},
)
    q = expected_demand(model, work, dealer_uid, t)
    length(prices) == length(q) ||
        error("foc_residual: prices length $(length(prices)) ≠ N_d $(length(q))")
    length(costs) == length(q) ||
        error("foc_residual: costs length $(length(costs)) ≠ N_d $(length(q))")
    J = share_matrix(model, work, params, dealer_uid, t)
    return q .+ J' * (prices .- costs)
end

# -------------------------------------------------------------------------------------------------
# Counterfactual: high-level wrapper bundling model + work + unpacked params + δ-decomposition
# outputs. apply!/undo! auto-refresh CCPs so callers never forget forward_pass!.
# -------------------------------------------------------------------------------------------------

mutable struct Counterfactual{TRegout, TFirstStage, THausman}
    model::DemandModel
    work::DemandWork
    δ::Vector{Float32}
    Πᵦ::Matrix{Float32}                         # Dn × K
    Πp::Vector{Float32}
    η::Vector{Float32}
    β_price::Float32
    regout::TRegout                                  # FixedEffectModels regression (loaded from JLD2)
    first_stage::TFirstStage
    J_t_lookup::Dict{QuarterKey, Int}
    hausman_fn::THausman
    dirty_blocks::BitVector                          # per-block stale flag; refresh! forward-passes only set bits
end

# Default Hausman lookup: looks up existing entries in model.product_info by (t, j, k, f).
# Returns NaN for new (t, j, f) triples the loader never saw — caller must pass a richer
# closure (e.g. one that calls `hausman_price` over a `mean_prices` GroupedDataFrame).
function default_hausman_fn(model::DemandModel)
    return function (t, j_minus_trim, k, f)
        j = (j_minus_trim[1], j_minus_trim[2], j_minus_trim[3], Float32(k))
        idx = get(model.product_lookup.tjf, (t, j, f), Int32(0))
        idx == 0 && return NaN
        return Float64(model.product_info[idx].p_hausman)
    end
end

"""
Counterfactual(model, params; decomp_path = "data/delta_decomposition_regout.jld2", hausman_fn = nothing)

Builds a counterfactual context around `model` at `params`, loading `regout` / `first_stage` /
`J_t_lookup` / `β_price` from the δ-decomposition JLD2 file. `apply!(cf, tweak)` and
`undo!(cf, handle)` then take no extra kwargs and auto-refresh the CCPs.
"""
function Counterfactual(
    model::DemandModel, params;
    decomp_path::AbstractString = "data/delta_decomposition_regout.jld2",
    hausman_fn::Union{Nothing, Function} = nothing,
)
    work = DemandWork(model)
    i = param_indices(model)
    δ = Vector{Float32}(params[i.i_δ])
    Πᵦ = reshape(Vector{Float32}(params[i.i_Πᵦ]), model.Dn, model.K)
    Πp = Vector{Float32}(params[i.i_Πp])
    η = Vector{Float32}(params[i.i_η])

    decomp = JLD2.load_object(decomp_path)
    regout = decomp.regout
    first_stage = decomp.first_stage
    df = decomp.df
    β_idx = findfirst(==("price"), coefnames(regout))
    isnothing(β_idx) && error("Counterfactual: no 'price' coefficient in regout")
    β_price = Float32(coef(regout)[β_idx])
    J_t_lookup = Dict{QuarterKey, Int}(
        (Int64(row.year), Int64(row.quarter)) => Int(row.J)
        for row in eachrow(unique(select(df, :year, :quarter, :J)))
    )
    hausman_fn_ = isnothing(hausman_fn) ? default_hausman_fn(model) : hausman_fn

    cf = Counterfactual(
        model, work, δ, Πᵦ, Πp, η,
        β_price, regout, first_stage, J_t_lookup, hausman_fn_,
        trues(model.n_blocks),
    )
    refresh!(cf)
    return cf
end

"""
refresh!(cf) — run forward_pass! over only the blocks marked dirty and clear them.
Called lazily by the read-only accessors (expected_demand / share_matrix / foc_residual).
"""
function refresh!(cf::Counterfactual)
    any(cf.dirty_blocks) || return cf
    blocks = findall(cf.dirty_blocks)
    forward_pass!(cf.work, cf.model, cf.δ, cf.Πᵦ, cf.Πp, cf.η; blocks)
    fill!(cf.dirty_blocks, false)
    return cf
end

# Mark every block touched by a handle dirty.
function _mark_dirty!(cf::Counterfactual, h::NamedTuple)
    if h.kind in (:price_multiply, :price_change)
        @inbounds for (bi, _) in h.dense_indices
            cf.dirty_blocks[bi] = true
        end
    elseif h.kind === :model_add || h.kind === :model_delete
        cf.dirty_blocks[h.bi] = true
    else
        fill!(cf.dirty_blocks, true)
    end
    return nothing
end

# High-level apply!/undo! — dispatch on tweak type, read kwargs from cf, mark dirty.

function apply!(cf::Counterfactual, tweak::PriceMultiply)
    h = apply!(tweak, cf.model, cf.δ; β_price = cf.β_price)
    ismissing(h) || _mark_dirty!(cf, h)
    return h
end

function apply!(cf::Counterfactual, tweak::PriceChange)
    h = apply!(tweak, cf.model, cf.δ; β_price = cf.β_price)
    ismissing(h) || _mark_dirty!(cf, h)
    return h
end

function apply!(cf::Counterfactual, tweak::ModelDelete)
    h = apply!(tweak, cf.model, cf.work, cf.δ)
    ismissing(h) || _mark_dirty!(cf, h)
    return h
end

function apply!(cf::Counterfactual, tweak::ModelAdd)
    h = apply!(
        tweak, cf.model, cf.work, cf.δ;
        regout = cf.regout, first_stage = cf.first_stage,
        J_t_lookup = cf.J_t_lookup, hausman_fn = cf.hausman_fn,
    )
    ismissing(h) || _mark_dirty!(cf, h)
    return h
end

function undo!(cf::Counterfactual, handle::NamedTuple)
    if handle.kind in (:price_multiply, :price_change)
        undo!(handle, cf.model, cf.δ)
    else
        undo!(handle, cf.model, cf.work, cf.δ)
    end
    _mark_dirty!(cf, handle)
    return cf
end

# High-level read-only accessors — lazily refresh dirty blocks before reading.

function expected_demand(cf::Counterfactual, dealer_uid::UUID, t::QuarterKey)
    refresh!(cf)
    return expected_demand(cf.model, cf.work, dealer_uid, t)
end

function share_matrix(cf::Counterfactual, dealer_uid::UUID, t::QuarterKey)
    refresh!(cf)
    params = _full_params(cf)
    return share_matrix(cf.model, cf.work, params, dealer_uid, t, mean_α = cf.β_price)
end

function foc_residual(
    cf::Counterfactual, dealer_uid::UUID, t::QuarterKey,
    prices::AbstractVector{Float32}, costs::AbstractVector{Float32},
)
    refresh!(cf)
    params = _full_params(cf)
    return foc_residual(cf.model, cf.work, params, dealer_uid, t, prices, costs)
end

# Rebuild a flat params vector from cf's unpacked pieces. Used by share_matrix / foc_residual
# which still take a `params` view in the low-level API (they only read i_Πp).
function _full_params(cf::Counterfactual)
    i = param_indices(cf.model)
    n_total = i.i_η[end]
    out = Vector{Float32}(undef, n_total)
    # δ may have grown past model.N (ModelAdd); the i.i_δ range only covers 1:N, which is fine
    # — share_matrix / foc_residual only read i_Πp anyway.
    @inbounds for k in 1:length(i.i_δ)
        out[i.i_δ[k]] = cf.δ[k]
    end
    Πᵦ_flat = vec(cf.Πᵦ)
    @inbounds for k in eachindex(i.i_Πᵦ)
        out[i.i_Πᵦ[k]] = Πᵦ_flat[k]
    end
    @inbounds for k in eachindex(i.i_Πp)
        out[i.i_Πp[k]] = cf.Πp[k]
    end
    out[i.i_η[1]] = cf.η[1]
    out[i.i_η[2]] = cf.η[2]
    return out
end
