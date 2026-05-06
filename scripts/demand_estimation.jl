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

function child_handle(r::LikelihoodRequest)
    return mapreduce(Demand.loglikelihood$(model, r.params), +, r.zs, init = 0.0f0)
end

function child_handle(r::GradientRequest)
    return mapreduce(
        Demand.dloglikelihood$(model, r.params), +, r.zs, init = zeros(Float32, length(r.params))
    )
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

            δ_init = copy(Float32.(@pull delta θ_tmp model))

            inner_result = optimize(
                OnceDifferentiable(
                    function (δ)
                        θ_tmp[i_δ] .= δ
                        return mpi_scatter(LikelihoodRequest, zips, θ_tmp)  # = -logL
                    end,
                    function (δ)
                        θ_tmp[i_δ] .= δ
                        g = mpi_scatter(GradientRequest, zips, θ_tmp)[i_δ]
                        @. g = ifelse(obs_count > 0.0f0, g, 0.0f0)  # freeze zero-obs products
                        return g
                    end,
                    δ_init,
                    inplace = false,
                ),
                δ_init,
                LBFGS(m = 10, linesearch = LineSearches.BackTracking()),
                Optim.Options(
                    iterations = 100,
                    g_tol = 1.0f-3,
                    allow_f_increases = true,
                    callback = state -> begin
                        if state.pseudo_iteration % 10 == 1
                            @info "\n  inner #$id iter $(state.pseudo_iteration): gnorm=$(round(norm(state.g_x); sigdigits=4))\n"
                        end
                        return false
                    end,
                ),
            )

            θ_tmp[i_δ] .= inner_result.minimizer
            _cached_π .= π
            _cached_θ_opt .= θ_tmp
            @info "\n  inner #$id: $(inner_result.iterations) L-BFGS iters, converged=$(Optim.converged(inner_result))\n"
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
