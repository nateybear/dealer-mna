using DealerMnA
import .Demand: DemandModel, FixedCoefType, initial_parameters, @indexof, @pull
using StatsBase: cov
using LinearAlgebra
LinearAlgebra.BLAS.set_num_threads(1)
using Optim
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

        # Inner problem: given non-linear params π, find δ*(π) = argmax_δ loglikelihood.
        function concentrated_θ!(π)
            π == _cached_π && return _cached_θ_opt
            θ_tmp = copy(θ)
            θ_tmp[i_nonδ] .= π
            δ₀ = Float32.(@pull delta θ_tmp model)
            inner_result = optimize(
                OnceDifferentiable(
                    function (δ)
                        θ_t = copy(θ_tmp)
                        (@pull delta θ_t model) .= δ
                        return mpi_scatter(LikelihoodRequest, zips, θ_t)
                    end,
                    function (δ)
                        θ_t = copy(θ_tmp)
                        (@pull delta θ_t model) .= δ
                        return mpi_scatter(GradientRequest, zips, θ_t)[i_δ]
                    end,
                    δ₀,
                    inplace = false,
                ),
                δ₀,
                LBFGS(m = 20),
                Optim.Options(
                    iterations = 500,
                    f_tol = 1.0f-9,
                    g_tol = 1.0f-6,
                    allow_f_increases = true,
                    show_trace = true,
                    show_every = 10,
                ),
            )
            (@pull delta θ_tmp model) .= inner_result.minimizer
            _cached_π .= π
            _cached_θ_opt .= θ_tmp
            @info "Inner: converged=$(Optim.converged(inner_result)), f=$(minimum(inner_result))"
            return _cached_θ_opt
        end

        # Outer problem: optimize likelihood + GMM over non-delta params, delta concentrated out.
        π₀ = Float32.(θ[i_nonδ])
        outer_result = optimize(
            OnceDifferentiable(
                function (π)
                    θ_opt = concentrated_θ!(π)
                    return mpi_scatter(LikelihoodRequest, zips, θ_opt) + Demand.Π̂(model, θ_opt, W)
                end,
                function (π)
                    θ_opt = concentrated_θ!(π)
                    return (mpi_scatter(GradientRequest, zips, θ_opt) .+ Demand.dΠ̂(model, θ_opt, W))[i_nonδ]
                end,
                π₀,
                inplace = false,
            ),
            π₀,
            LBFGS(m = 20),
            Optim.Options(
                store_trace = true,
                show_trace = true,
                show_every = 1,
                iterations = 5000,
                f_tol = 1.0f-9,
                g_tol = 1.0f-6,
                allow_f_increases = true,
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
