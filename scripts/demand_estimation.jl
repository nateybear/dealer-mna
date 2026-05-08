using DealerMnA
import .Demand: DemandModel, FixedCoefType, initial_parameters, @indexof, @pull
using StatsBase: cov
using LinearAlgebra
LinearAlgebra.BLAS.set_num_threads(1)
using Optim
import LineSearches
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
    return mapreduce(Demand.loglikelihood$(model, r.params), +, r.zs, init = 0.0f0)
end

function child_handle(r::GradientRequest)
    return mapreduce(
        Demand.dloglikelihood$(model, r.params), +, r.zs, init = zeros(Float32, length(r.params))
    )
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
        zips = Registrations.zips().zip_code
        i_δ = @indexof delta θ model
        i_nonδ = setdiff(1:length(θ), i_δ)
        # obs_count: used to freeze zero-obs products in the inner δ solve.
        obs_count = Demand.δ_obs_counts(model)

        # Cache: avoid re-running the inner problem when f and g are called at the same π.
        _cached_π = fill(NaN32, length(i_nonδ))
        _cached_θ_opt = copy(θ)
        _inner_iter = Ref(0)
        _last_ll = Ref(NaN32)
        _last_gmm = Ref(NaN32)

        ls = HagerZhang()

        # Inner problem: given non-delta params π, find δ*(π) = argmax_δ loglikelihood.
        # Uses L-BFGS over δ only. The BLP contraction (SQUAREM) has an asymptotically
        # slow modulus (~0.9999/step) for this aggregate shared-δ formulation because the
        # second difference v ≈ 0 (near-linear convergence), leaving SQUAREM ineffective.
        # L-BFGS builds a curvature approximation and converges superlinearly.
        # Zero-obs products are frozen by zeroing their gradient (same semantics as BLP).
        function concentrated_θ!(π)
            if π == _cached_π
                return _cached_θ_opt
            end
            _inner_iter[] += 1
            id = _inner_iter[]

            θ_tmp = copy(θ)
            θ_tmp[i_nonδ] .= π
            # warm start: reuse last δ* rather than resetting to initial
            (@pull delta θ_tmp model) .= @pull delta _cached_θ_opt model

            newton_iter = 0
            while true
                newton_iter += 1
                δ_cur = @pull delta θ_tmp model
                g_full = mpi_scatter(GradientRequest, zips, θ_tmp)  # gradient of -ℓ
                g = @pull delta g_full model                          # δ-block only
                gnorm = norm(g)
                if gnorm ≤ 1.0f-4
                    @info "  inner #$id newton: converged in $newton_iter iters (gnorm=$gnorm)"
                    break
                end
                # d = (-H_ℓ)^{-1} ∇_δ ℓ: Newton ascent direction for ℓ (descent for -ℓ).
                # mpi_newton_delta_direction expects ∇_δ ℓ (not negated), so pass -g.
                d = mpi_newton_delta_direction(θ_tmp, zips, -g)
                # Precompute ϕ0 and dϕ0 from values already in hand.
                ϕ0 = mpi_scatter(LikelihoodRequest, zips, θ_tmp)  # -ℓ at current point
                dϕ0 = dot(g, d)  # ∇(-ℓ)·d = g·d ≤ 0 (valid descent direction)
                function ϕ(α)
                    θ_tmp_tmp = copy(θ_tmp)
                    (@pull delta θ_tmp_tmp model) .= δ_cur + α * d
                    return mpi_scatter(LikelihoodRequest, zips, θ_tmp_tmp)
                end
                function dϕ(α)
                    θ_tmp_tmp = copy(θ_tmp)
                    (@pull delta θ_tmp_tmp model) .= δ_cur + α * d
                    g_trial = mpi_scatter(GradientRequest, zips, θ_tmp_tmp)
                    return dot(@pull(delta, g_trial, model), d)  # ∇(-ℓ)·d at trial point
                end
                ϕdϕ(α) = (ϕ(α), dϕ(α))
                alpha, phi_alpha, _, _ = ls(ϕ, dϕ, ϕdϕ, 1.0f0, ϕ0, dϕ0)
                @info "  inner #$id newton iter $newton_iter: -logL=$(round(phi_alpha; sigdigits=6)) gnorm=$(round(gnorm; sigdigits=4)) α=$(round(alpha; sigdigits=4))"
                δ_cur .+= alpha * d
            end
            _cached_π .= π
            _cached_θ_opt .= θ_tmp
            return _cached_θ_opt
        end

        function outer_optimize(π₀, W_cur, label)
            return optimize(
                OnceDifferentiable(
                    function (π)
                        θ_opt = concentrated_θ!(π)
                        ll = mpi_scatter(LikelihoodRequest, zips, θ_opt)
                        gmm = Demand.Π̂(model, θ_opt, W_cur)
                        _last_ll[] = ll
                        _last_gmm[] = gmm
                        return ll + gmm
                    end,
                    function (π)
                        θ_opt = concentrated_θ!(π)
                        g = (mpi_scatter(GradientRequest, zips, θ_opt) .+ Demand.dΠ̂(
                            model, θ_opt, W_cur
                        ))[i_nonδ]
                        gn = norm(g)
                        return gn > 1.0f4 ? g .* (1.0f4 / gn) : g
                    end,
                    π₀,
                    inplace = false,
                ),
                π₀,
                LBFGS(m = 20, linesearch = LineSearches.BackTracking()),
                Optim.Options(
                    store_trace = true,
                    show_trace = false,
                    iterations = 5000,
                    g_tol = 1.0f-3,
                    f_reltol = NaN,
                    f_abstol = NaN,
                    x_reltol = NaN,
                    x_abstol = NaN,
                    allow_f_increases = true,
                    callback = state -> begin
                        @info "\n$label iter $(state.pseudo_iteration): " *
                            "total=$(round(state.f_x; sigdigits=6)) " *
                            "-logL=$(round(_last_ll[]; sigdigits=6)) " *
                            "Π̂=$(round(_last_gmm[]; sigdigits=6)) " *
                            "gnorm=$(round(norm(state.g_x); sigdigits=4)) " *
                            "inner_iters=$(_inner_iter[])\n"
                        return false
                    end,
                ),
            )
        end

        # Step 1: use identity weighting — optimalW at initial θ is degenerate
        # (ξ̂ has near-zero variance at initialization, making cov(m) singular).
        # Efficient W is only meaningful at a consistent first-step estimate.
        W = I
        π₀ = Float32.(θ[i_nonδ])
        result1 = outer_optimize(π₀, W, "step1")
        θ̂₁ = copy(_cached_θ_opt)
        θ̂₁[i_nonδ] .= result1.minimizer
        @info "\nStep 1 done: total=$(round(result1.minimum; sigdigits=6)) converged=$(Optim.converged(result1))\n"

        # Step 2: update W at θ̂₁ and re-optimize for efficiency.
        W_opt = Demand.optimalW(model, θ̂₁)
        @info "\nRecomputed optimal W for step 2\n"
        _inner_iter[] = 0
        _cached_π .= NaN32  # force inner re-solve at new W
        result2 = outer_optimize(result1.minimizer, W_opt, "step2")
        θ̂₂ = copy(_cached_θ_opt)
        θ̂₂[i_nonδ] .= result2.minimizer
        @info "\nStep 2 done: total=$(round(result2.minimum; sigdigits=6)) converged=$(Optim.converged(result2))\n"

        JLD2.save_object(
            snakemake.output[1],
            (; minimizer = θ̂₂, step1 = result1, step2 = result2),
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
                resp = child_handle(msg)
                MPI.send(resp, 0, 0, comm)
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
