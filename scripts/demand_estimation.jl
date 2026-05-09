using DealerMnA
import .Demand: DemandModel, FixedCoefType, initial_parameters, @indexof, @pull
using StatsBase: cov
using LinearAlgebra
LinearAlgebra.BLAS.set_num_threads(1)
using Optim
import Random: seed!
import LineSearches: HagerZhang

seed!(78704)

snakemake!()

using MPI
MPI.Init()

model = DemandModel{FixedCoefType}(; JLD2.load_object(snakemake.input["model"])...)

# Sorted so parent and child agree on iteration order regardless of Dict hashing.
const T_BLOCK_ORDER = sort(collect(pairs(Demand.δ_t_blocks(model))), by = first)

comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
syze = MPI.Comm_size(comm)

MPI.Barrier(comm)

struct LikelihoodRequest
    params::Vector{Float32}
    zs::Vector{String}
end

struct GradientRequest
    params::Vector{Float32}
    zs::Vector{String}
end

struct StopSignal end

struct CCPsRequest
    params::Vector{Float32}
    zips::Vector{String}
end

struct CombinedRequest
    params::Vector{Float32}
    zips::Vector{String}
end

function mpi_scatter(T, zips, params)
    chunksize = cld(length(zips), syze - 1)
    maxrank = 0 # might not use all of the ranks in edge cases...
    for (i, chunk) in enumerate(Iterators.partition(zips, chunksize))
        MPI.send(T(params, chunk), i, 0, comm)
        maxrank = i
    end
    out = T === LikelihoodRequest ? 0.0f0 : zeros(Float32, length(params))
    for i in 1:maxrank
        msg, _ = MPI.recv(i, 0, comm)
        if msg isa StopSignal
            @info "Process $i signaled to stop"
            stop!()
            error("Stopping optimization as requested by process $i")
        else
            out -= msg
        end
    end
    return out
end

function child_handle(r::LikelihoodRequest)
    resp = mapreduce(+, r.zs, init = 0.0f0) do z
        Demand.loglikelihood(model, r.params, z)
    end
    MPI.send(resp, 0, 0, comm)
    return nothing
end

function child_handle(r::GradientRequest)
    resp = mapreduce(+, r.zs, init = zeros(Float32, length(r.params))) do z
        Demand.dloglikelihood(model, r.params, z)
    end
    MPI.send(resp, 0, 0, comm)
    return nothing
end

function child_handle(r::CCPsRequest)
    all_idxs = Demand.δ_indices(model)
    Z_local = length(r.zips)
    S = length(model.u.draws)
    inv_S = 1.0f0 / Float32(S)

    # Averaged CCPs for each assigned zip: one row per δ group, one column per zip.
    all_ccps = Matrix{Float32}(undef, length(all_idxs), Z_local)
    for (zi, z) in enumerate(r.zips)
        ccp_S = Demand.CCPs(model, r.params, z)  # S × T × J × F
        @inbounds for (i, idx) in enumerate(all_idxs)
            s = 0.0f0
            for si in 1:S
                s += ccp_S[si, Tuple(idx)...]
            end
            all_ccps[i, zi] = s * inv_S
        end
    end

    # Send each time-block slice; tag = block index (1-based) so parent can receive in order.
    for (bi, (_, i_δs)) in enumerate(T_BLOCK_ORDER)
        MPI.send(all_ccps[i_δs, :], 0, bi, comm)  # N_t × Z_local
    end
end

function child_handle(r::CombinedRequest)
    all_idxs = Demand.δ_indices(model)
    Z_local = length(r.zips)
    ll_sum = 0.0f0
    grad_sum = zeros(Float32, length(r.params))
    all_ccps = Matrix{Float32}(undef, length(all_idxs), Z_local)
    for (zi, z) in enumerate(r.zips)
        ll_z, grad_z, ccp_vec = Demand.loglikelihood_gradient_and_ccps(model, r.params, z)
        ll_sum += ll_z
        grad_sum .+= grad_z
        all_ccps[:, zi] .= ccp_vec
    end
    MPI.send((grad_sum, ll_sum), 0, 0, comm)
    for (bi, (_, i_δs)) in enumerate(T_BLOCK_ORDER)
        MPI.send(all_ccps[i_δs, :], 0, bi, comm)
    end
    return nothing
end

# Woodbury Newton step for one time block.
# -H_t = diag(d) - U*diag(M_z)*U'  where d = U*M_z  (positive semidefinite block Hessian of -ℓ).
# g_t:  N_t loglikelihood gradient for this block's δ parameters.
# Returns (-∂²ℓ/∂δ_t²)^{-1} g_t — Newton ascent direction for ℓ (negate for descent on -ℓ).
function woodbury_newton_step(
    U::Matrix{Float32},    # N_t × Z: per-zip averaged CCPs for products in this block
    M_z::Vector{Float32},  # Z: market sizes in the same column order as U
    g_t::Vector{Float32},  # N_t: loglikelihood gradient for this block
)::Vector{Float32}
    d = U * M_z                                    # N_t: diagonal of -H_t = D
    inv_d = 1.0f0 ./ max.(d, 1.0f-12)
    v = inv_d .* g_t                               # v = D⁻¹g
    DinvU = inv_d .* U                             # D⁻¹U, N_t × Z
    middle = Diagonal(1.0f0 ./ M_z) - U' * DinvU  # M⁻¹ - U'D⁻¹U, Z × Z
    w = Symmetric(middle) \ (U' * v)               # (M⁻¹ - U'D⁻¹U)⁻¹ U'v, Z-vector
    return v + DinvU * w                           # = (-H_t)⁻¹ g_t via Woodbury
end

# Scatter a CCPsRequest, collect per-block CCP matrices, and return the block-wise
# Woodbury Newton ascent direction for all δ parameters.
# g_δ: loglikelihood gradient for δ (length n_delta), NOT negated.
# Returns (-∂²ℓ/∂δ²)^{-1} g_δ; negate this to get the descent direction for -ℓ.
function mpi_newton_delta_direction(
    params::Vector{Float32},
    zips::Vector{String},
    g_δ::Vector{Float32},
)::Vector{Float32}
    chunksize = cld(length(zips), syze - 1)
    zip_chunks = collect(Iterators.partition(zips, chunksize))
    maxrank = length(zip_chunks)
    for (i, chunk) in enumerate(zip_chunks)
        MPI.send(CCPsRequest(params, collect(chunk)), i, 0, comm)
    end
    # M_z in the same order as zips (which matches the concatenated chunk columns in U)
    M_zs = Float32[model.agent_data[z].M for z in zips]

    step = Vector{Float32}(undef, length(g_δ))
    for (bi, (_, i_δs)) in enumerate(T_BLOCK_ORDER)
        N_t = length(i_δs)
        U = Matrix{Float32}(undef, N_t, length(zips))
        col = 1
        for i in 1:maxrank
            ccp_block, _ = MPI.recv(i, bi, comm)  # N_t × Z_child
            ncols = size(ccp_block, 2)
            U[:, col:(col + ncols - 1)] .= ccp_block
            col += ncols
        end
        step[i_δs] .= woodbury_newton_step(U, M_zs, g_δ[i_δs])
    end
    return step
end

# Scatter a CombinedRequest; receive (grad, ll) on tag 0 from all workers, then assemble
# per-block CCP matrices and compute Woodbury Newton step.
# Returns (-∇ℓ, Newton ascent step for ℓ over δ, -ℓ) — matching mpi_scatter sign convention.
function mpi_combined_newton_step(
    params::Vector{Float32},
    zips::Vector{String},
)::Tuple{Vector{Float32}, Vector{Float32}, Float32}
    chunksize = cld(length(zips), syze - 1)
    zip_chunks = collect(Iterators.partition(zips, chunksize))
    maxrank = length(zip_chunks)
    for (i, chunk) in enumerate(zip_chunks)
        MPI.send(CombinedRequest(params, collect(chunk)), i, 0, comm)
    end
    M_zs = Float32[model.agent_data[z].M for z in zips]
    g_full = zeros(Float32, length(params))  # accumulates +∇ℓ
    ll_sum = 0.0f0                           # accumulates +ℓ
    for i in 1:maxrank
        (grad_i, ll_i), _ = MPI.recv(i, 0, comm)
        g_full .+= grad_i
        ll_sum += ll_i
    end
    i_δ_all = @indexof delta params model
    g_δ = g_full[i_δ_all]  # +∇_δ ℓ
    step = Vector{Float32}(undef, length(g_δ))
    for (bi, (_, i_δs)) in enumerate(T_BLOCK_ORDER)
        N_t = length(i_δs)
        U = Matrix{Float32}(undef, N_t, length(zips))
        col = 1
        for i in 1:maxrank
            ccp_block, _ = MPI.recv(i, bi, comm)
            ncols = size(ccp_block, 2)
            U[:, col:(col + ncols - 1)] .= ccp_block
            col += ncols
        end
        step[i_δs] .= woodbury_newton_step(U, M_zs, g_δ[i_δs])
    end
    return -g_full, step, -ll_sum
end

stop!() =
    for i in 1:(syze - 1)
        MPI.send(StopSignal(), i, 0, comm)
    end

with_logging() do
    if rank == 0
        @info "Starting optimization with $(syze) processes"
        θ =
            if haskey(snakemake.input, "params_restart") &&
                !isempty(snakemake.input["params_restart"]) &&
                isfile(snakemake.input["params_restart"])
                JLD2.load_object(snakemake.input["params_restart"]).minimizer
            else
                initial_parameters(model)
            end
        @info "Initial parameters set"
        zips = Registrations.zips().zip_code
        i_δ = @indexof delta θ model
        i_nonδ = setdiff(1:length(θ), i_δ)

        # Cache: avoid re-running the inner problem when f, g!, h! are called at the same π.
        _cached_π = fill(NaN32, length(i_nonδ))
        _cached_θ_opt = copy(θ)
        _inner_iter = Ref(0)
        _last_ll = Ref(NaN32)

        ls = HagerZhang()

        # Inner problem: given non-delta params π, find δ*(π) = argmax_δ loglikelihood.
        # Custom block-diagonal Newton: solves (-H_t)*Δδ_t = g_t per time block, never
        # forming the full n_delta×n_delta Hessian. Each MPI round returns the full
        # block-diagonal H_flat (1.25 GB on workers) accumulated in-place to avoid
        # per-zip allocations. Quadratic convergence in 2–5 iterations.
        function concentrated_θ!(π)
            if π == _cached_π
                return _cached_θ_opt
            end
            _inner_iter[] += 1
            @info "Starting inner optimization #$(_inner_iter[])"
            id = _inner_iter[]

            θ_tmp = copy(θ)
            θ_tmp[i_nonδ] .= π
            # warm start: reuse last δ* rather than resetting to initial
            (@pull delta θ_tmp model) .= @pull delta _cached_θ_opt model

            newton_iter = 0
            while true
                newton_iter += 1
                δ_cur = @pull delta θ_tmp model
                # Combined round: gradient, Newton step, and ϕ0 in one MPI scatter.
                t_combined = @elapsed begin
                    g_full, d, ϕ0 = mpi_combined_newton_step(θ_tmp, zips)
                end
                g = @pull delta g_full model  # -∇_δ ℓ
                gnorm = norm(g)
                if gnorm ≤ 1.0f-3 || isapprox(ϕ0, 0) 
                    @info "  inner #$id newton: converged in $newton_iter iters (gnorm=$gnorm)"
                    break
                end
                dϕ0 = dot(g, d)  # ∇(-ℓ)·d ≤ 0 (valid descent direction)
                @info "  inner #$id newton iter $newton_iter: gnorm=$(round(gnorm; sigdigits=4)) dϕ0=$(round(dϕ0; sigdigits=4)) combined_time=$(round(t_combined; sigdigits=3))s"
                function ϕ(α)
                    θ_tmp_tmp = copy(θ_tmp)
                    (@pull delta θ_tmp_tmp model) .= δ_cur + α * d
                    return mpi_scatter(LikelihoodRequest, zips, θ_tmp_tmp)
                end
                function dϕ(α)
                    θ_tmp_tmp = copy(θ_tmp)
                    (@pull delta θ_tmp_tmp model) .= δ_cur + α * d
                    g_trial = mpi_scatter(GradientRequest, zips, θ_tmp_tmp)
                    return dot(@pull(delta, g_trial, model), d)
                end
                ϕdϕ(α) = (ϕ(α), dϕ(α))
                alpha, phi_alpha = ls(ϕ, dϕ, ϕdϕ, 1.0f0, ϕ0, dϕ0)
                @info "  inner #$id newton iter $newton_iter: -logL=$(round(phi_alpha; sigdigits=6)) α=$(round(alpha; sigdigits=4))"
                δ_cur .+= alpha * d
            end
            _cached_π .= π
            _cached_θ_opt .= θ_tmp
            return _cached_θ_opt
        end

        function outer_optimize(π₀, label)
            function outer_f(π)
                θ_opt = concentrated_θ!(π)
                ll = mpi_scatter(LikelihoodRequest, zips, θ_opt)
                _last_ll[] = ll
                return ll
            end

            function outer_g!(G, π)
                θ_opt = concentrated_θ!(π)
                G .= mpi_scatter(GradientRequest, zips, θ_opt)[i_nonδ]
                return nothing
            end

            return optimize(
                OnceDifferentiable(outer_f, outer_g!, π₀),
                π₀,
                LBFGS(),
                Optim.Options(
                    store_trace = true,
                    show_trace = false,
                    iterations = 500,
                    g_tol = 1.0f-3,
                    callback = state -> begin
                        @info "\n$label iter $(state.pseudo_iteration): " *
                            "total=$(round(state.f_x; sigdigits=6)) " *
                            "-logL=$(round(_last_ll[]; sigdigits=6)) " *
                            "gnorm=$(round(norm(state.g_x); sigdigits=4)) " *
                            "inner_iters=$(_inner_iter[])\n"
                        return false
                    end,
                ),
            )
        end

        π₀ = Float32.(θ[i_nonδ])
        result = outer_optimize(π₀, "step1")
        @info "\nEstimation done: total=$(round(result.minimum; sigdigits=6)) converged=$(Optim.converged(result))\n"

        JLD2.save_object(
            snakemake.output[1],
            result,
        )
        stop!()
    else
        while true
            msg, _ = MPI.recv(0, 0, comm)
            if msg isa StopSignal
                @info "Received stop signal from master"
                break
            end
            try
                child_handle(msg)
            catch e
                @error "Error in worker process $rank: $e"
                MPI.send(StopSignal(), 0, 0, comm)
                break
            end
        end
    end
end

MPI.Barrier(comm)
MPI.Finalize()
