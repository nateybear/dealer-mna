using DealerMnA
import .Demand: DemandModel, FixedCoefType, initial_parameters
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

function parent_scatter(T, zips, W, params)
    chunksize = cld(length(zips), syze - 1)
    @info "Partitioning into chunks of size $chunksize"
    maxrank = 0 # might not use all of the ranks in edge cases...
    for (i, chunk) in enumerate(Iterators.partition(zips, chunksize))
        MPI.send(T(params, chunk), i, 0, comm)
        maxrank = i
    end
    out = parent_handle(T, model, W, params)
    @info
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

        # Two-step GMM for the combined objective; one pass for likelihood.
        T = snakemake.params["objective"] == "likelihood" ? 1 : 2
        i = 0
        while i < T
            W = Demand.optimalW(model, θ)
            f = OnceDifferentiable(
                parent_scatter$(LikelihoodRequest, zips, W),
                parent_scatter$(GradientRequest, zips, W),
                θ,
                inplace = false,
            )
            i += 1
            @info "In loop $i"
            result = optimize(
                f,
                θ,
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
            JLD2.save_object(snakemake.output[1], result)
            p, ext = splitext(snakemake.output[1])
            obj = snakemake.params["objective"]
            JLD2.save_object("$(p)_$(obj)_$(i)$(ext)", result)
            θ = result.minimizer
        end
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
