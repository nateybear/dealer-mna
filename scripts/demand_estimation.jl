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
    @info "Partitioning into chunks of size $chunksize"
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

function parent_scatter(T, zips, W, params)
    return mpi_scatter(T, zips, params) + parent_handle(T, model, W, params)
end

function parent_handle(::Type{LikelihoodRequest}, model, W, params)
    islikelihood = snakemake.params["objective"] == "likelihood"
    if islikelihood
        return 0.0f0
    else
        # GMM moment objective is global (no zip dimension); evaluated once
        return Demand.Π̂(model, params, W)
    end
end

function parent_handle(::Type{GradientRequest}, model, W, params)
    islikelihood = snakemake.params["objective"] == "likelihood"
    if islikelihood
        return zeros(Float32, length(params))
    else
        return Demand.dΠ̂(model, params, W)
    end
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
        p, ext = splitext(snakemake.output[1])

        # Grieco et al. cheap estimator: concentrate out delta via likelihood in the inner
        # problem, then optimize the full objective over the remaining non-linear params.
        i_δ = @indexof delta θ model
        i_nonδ = setdiff(1:length(θ), i_δ)
        W = Demand.optimalW(model, θ)

        # Cache avoids re-running the inner optimization for f and g at the same outer π.
        _cached_π = fill(NaN32, length(i_nonδ))
        _cached_θ_opt = copy(θ)

        # Counters shared across closures for log correlation.
        _inner_call = Ref(0)
        _last_ll = Ref(NaN32)
        _last_gmm = Ref(NaN32)

        # Inner problem: given non-linear params π, find δ*(π) = argmax_δ loglikelihood.
        # Uses Barzilai–Borwein gradient descent: one gradient scatter per iteration,
        # no line search. BB estimates a scalar step from consecutive gradient differences,
        # giving adaptive curvature information without the overhead of a full line search.
        function concentrated_θ!(π)
            if π == _cached_π
                @info "  inner #$(_inner_call[]): cache hit"
                return _cached_θ_opt
            end
            _inner_call[] += 1
            id = _inner_call[]
            @info "  inner #$id: starting BB ($(length(i_δ)) delta params)"

            θ_tmp = copy(θ)
            θ_tmp[i_nonδ] .= π
            (@pull delta θ_tmp model) .= @pull delta _cached_θ_opt model

            g_prev = zeros(Float32, length(i_δ))
            δ_prev = copy(Float32.(@pull delta θ_tmp model))
            α = 1.0f-3  # conservative initial step; BB updates it after the first iteration
            gnorm = Inf32
            iter = 0

            for iter in 1:500
                g = mpi_scatter(GradientRequest, zips, θ_tmp)[i_δ]  # gradient of -logL
                gnorm = maximum(abs, g)
                @info "  inner #$id BB $iter: gnorm=$(round(gnorm; sigdigits=4)) α=$(round(α; sigdigits=4))"
                gnorm < 1.0f-4 && break

                if iter > 1
                    s = (@pull delta θ_tmp model) .- δ_prev  # δ_k - δ_{k-1}
                    y = g .- g_prev                           # ∇f_k - ∇f_{k-1}
                    sy = dot(s, y)
                    # BB2 step: ||y||² / (s'y) — tends to take larger, more aggressive steps
                    sy > 0 && (α = clamp(dot(y, y) / sy, 1.0f-8, 1.0f4))
                end

                δ_prev .= @pull delta θ_tmp model
                g_prev .= g
                (@pull delta θ_tmp model) .-= α .* g
            end

            @info "  inner #$id: done — iters=$iter converged=$(gnorm < 1.0f-4) final_gnorm=$(round(gnorm; sigdigits=4))"
            _cached_π .= π
            _cached_θ_opt .= θ_tmp
            return _cached_θ_opt
        end

        # Outer problem: optimize likelihood + GMM over non-delta params, delta concentrated out.
        π₀ = Float32.(θ[i_nonδ])
        outer_result = optimize(
            OnceDifferentiable(
                function (π)
                    θ_opt = concentrated_θ!(π)
                    ll = mpi_scatter(LikelihoodRequest, zips, θ_opt)
                    gmm = Demand.Π̂(model, θ_opt, W)
                    _last_ll[] = ll
                    _last_gmm[] = gmm
                    return ll + gmm
                end,
                function (π)
                    θ_opt = concentrated_θ!(π)
                    return (mpi_scatter(GradientRequest, zips, θ_opt) .+ Demand.dΠ̂(model, θ_opt, W))[i_nonδ]
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
                f_tol = 1.0f-9,
                g_tol = 1.0f-6,
                allow_f_increases = true,
                callback = state -> begin
                    @info "outer iter $(state.iteration): " *
                          "total=$(round(state.value; sigdigits=6)) " *
                          "-logL=$(round(_last_ll[]; sigdigits=6)) " *
                          "Π̂=$(round(_last_gmm[]; sigdigits=6)) " *
                          "gnorm=$(round(state.g_norm; sigdigits=4)) " *
                          "inner_calls=$(_inner_call[])"
                    return false
                end,
            ),
        )

        # Reconstruct the full optimal parameter vector and save for restart compatibility.
        θ_star = copy(_cached_θ_opt)
        θ_star[i_nonδ] .= outer_result.minimizer
        JLD2.save_object(snakemake.output[1], (; minimizer = θ_star, outer = outer_result))
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
