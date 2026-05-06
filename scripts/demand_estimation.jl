using DealerMnA
import .Demand: DemandModel, FixedCoefType, initial_parameters, @indexof, @pull
using StatsBase: cov
using LinearAlgebra
LinearAlgebra.BLAS.set_num_threads(1)
using Optim
import LineSearches
import Random: seed!

seed!(78704)

snakemake!()

using MPI
MPI.Init()

model = DemandModel{FixedCoefType}(; JLD2.load_object(snakemake.input["model"])...)

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

struct NewtonRequest
    params::Vector{Float32}
    zs::Vector{String}
end

struct StopSignal end

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

# Returns (g_δ, H_flat) summed over all zips: ∇_δ logL and block-diagonal Hessian of logL.
# Both are positive-logL-convention (g_δ > 0 means increase δ; H_flat is neg-def).
function mpi_scatter_newton(zips, params)
    chunksize = cld(length(zips), syze - 1)
    maxrank = 0
    for (i, chunk) in enumerate(Iterators.partition(zips, chunksize))
        MPI.send(NewtonRequest(params, collect(chunk)), i, 0, comm)
        maxrank = i
    end
    n_groups = Demand.n_delta(model)
    n_hess = Demand.hessian_δ_storage(model)
    g_total = zeros(Float32, n_groups)
    H_total = zeros(Float32, n_hess)
    for i in 1:maxrank
        msg, _ = MPI.recv(i, 0, comm)
        if msg isa StopSignal
            @info "Process $i signaled to stop"
            stop!()
            error("Stopping optimization as requested by process $i")
        else
            g_i, H_i = msg
            g_total .+= g_i
            H_total .+= H_i
        end
    end
    return g_total, H_total
end

function child_handle(r::LikelihoodRequest)
    return mapreduce(Demand.loglikelihood$(model, r.params), +, r.zs, init = 0.0f0)
end

function child_handle(r::GradientRequest)
    return mapreduce(
        Demand.dloglikelihood$(model, r.params), +, r.zs, init = zeros(Float32, length(r.params))
    )
end

function child_handle(r::NewtonRequest)
    n_groups = Demand.n_delta(model)
    n_hess = Demand.hessian_δ_storage(model)
    g_acc = zeros(Float32, n_groups)
    H_acc = zeros(Float32, n_hess)
    for z in r.zs
        g_z, H_z = Demand.δ_grad_and_hessian(model, r.params, z)
        g_acc .+= g_z
        H_acc .+= H_z
    end
    return (g_acc, H_acc)
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

        n_groups = length(i_δ)
        t_blocks = Demand.δ_t_blocks(model)
        offsets = Demand.hessian_δ_offsets(model)

        n_hess = Demand.hessian_δ_storage(model)
        @info "Block-diagonal Hessian: $n_hess Float32 elements per worker ($(round(n_hess * 4 / 1e6; sigdigits = 3)) MB); dense inner H is $(round(n_groups^2 * 4 / 1e6; sigdigits = 3)) MB"

        # Cache: avoid re-running the inner problem when f and g are called at the same π.
        _cached_π = fill(NaN32, length(i_nonδ))
        _cached_θ_opt = copy(θ)
        _inner_iter = Ref(0)
        _last_ll = Ref(NaN32)
        _last_gmm = Ref(NaN32)

        # Inner Newton cache: g! and h! share one MPI round when called at the same δ.
        _newton_δ_cache = fill(NaN32, n_groups)
        _newton_g_cache = zeros(Float32, n_groups)
        _newton_H_flat_cache = zeros(Float32, n_hess)

        # Inner problem: given non-delta params π, find δ*(π) = argmax_δ loglikelihood.
        # Uses Newton's method (TwiceDifferentiable + NewtonTrustRegion) with the exact
        # block-diagonal Hessian of logL w.r.t. δ. Quadratic convergence in 2–5 iterations.
        # Zero-obs products are frozen by regularizing their Hessian diagonal to 1.0 and
        # zeroing their gradient, making the Newton step a no-op for those coordinates.
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

            δ_init = copy(Float32.(@pull delta θ_tmp model))

            # Ensure gradient and Hessian are computed together for a given δ.
            function _ensure_newton_cache!(δ)
                if δ == _newton_δ_cache
                    return
                end
                θ_tmp[i_δ] .= δ
                g_raw, H_raw = mpi_scatter_newton(zips, θ_tmp)
                @. g_raw = ifelse(obs_count > 0.0f0, g_raw, 0.0f0)
                _newton_g_cache .= .-g_raw  # gradient of -logL (minimization convention)
                _newton_H_flat_cache .= H_raw
                _newton_δ_cache .= δ
                return nothing
            end

            function inner_g!(G, δ)
                _ensure_newton_cache!(δ)
                G .= _newton_g_cache
                return nothing
            end

            # Hessian of -logL (positive definite). Block-diagonal by t; filled from H_flat.
            # Zero-obs products get H[i,i]=1, H[i,:]=H[:,i]=0 so Newton ignores them.
            function inner_h!(H, δ)
                _ensure_newton_cache!(δ)
                fill!(H, 0.0f0)
                for (t, grp_idxs) in t_blocks
                    J_t = length(grp_idxs)
                    off = offsets[t]
                    H_t = reshape(_newton_H_flat_cache[off:(off + J_t^2 - 1)], J_t, J_t)
                    H[grp_idxs, grp_idxs] .= .-H_t  # -H_logL is positive definite
                end
                for (i, cnt) in enumerate(obs_count)
                    if cnt == 0.0f0
                        @view(H[i, :]) .= 0.0f0
                        @view(H[:, i]) .= 0.0f0
                        H[i, i] = 1.0f0
                    end
                end
                return nothing
            end

            inner_result = optimize(
                TwiceDifferentiable(
                    function (δ)
                        θ_tmp[i_δ] .= δ
                        return mpi_scatter(LikelihoodRequest, zips, θ_tmp)
                    end,
                    inner_g!,
                    inner_h!,
                    δ_init,
                ),
                δ_init,
                NewtonTrustRegion(),
                Optim.Options(
                    iterations = 30,
                    g_tol = 1.0f-3,
                    callback = state -> begin
                        if state.iteration % 5 == 1
                            @info "\n  inner #$id iter $(state.iteration): gnorm=$(round(norm(state.g_x); sigdigits = 4))\n"
                        end
                        return false
                    end,
                ),
            )

            θ_tmp[i_δ] .= inner_result.minimizer
            _cached_π .= π
            _cached_θ_opt .= θ_tmp
            @info "\n  inner #$id: $(inner_result.iterations) Newton iters, converged=$(Optim.converged(inner_result))\n"
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
