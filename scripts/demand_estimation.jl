using DealerMnA
import .Demand: DemandModel, FixedCoefType, glorot_initialization
using StatsBase: cov
using LinearAlgebra
LinearAlgebra.BLAS.set_num_threads(1)
using Optim
import Random: shuffle!, seed!

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
    z::String
end

struct GradientRequest
    params::Vector{Float32}
    z::String
end

struct StopSignal end

function parent_scatter(T, zips, W, params)
    shuffle!(zips)
    for i in 1:(syze - 1)
        MPI.send(T(params, zips[i]), i, 0, comm)
    end
    out = parent_handle(T, model, W, params)
    for i in 1:(syze - 1)
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

parent_handle(::Type{LikelihoodRequest}, model, W, params) = Demand.Π̂(model, params, W)
parent_handle(::Type{GradientRequest}, model, W, params) = Demand.dΠ̂(model, params, W)

child_handle(r::LikelihoodRequest) = Demand.loglikelihood(model, r.params, r.z)
child_handle(r::GradientRequest) = Demand.dloglikelihood(model, r.params, r.z)

stop!() =
    for i in 1:(syze - 1)
        MPI.send(StopSignal(), i, 0, comm)
    end

with_logging() do
    if rank == 0
        @info "Starting optimization with $(syze) processes"
        θ = glorot_initialization(model)
        W = Demand.optimalW(model, θ)
        zips = Registrations.zips().zip_code
        f = OnceDifferentiable(
            parent_scatter$(LikelihoodRequest, zips, W),
            parent_scatter$(GradientRequest, zips, W),
            θ,
            inplace = false,
        )
        result = optimize(
            f,
            θ,
            Adam(),
            Optim.Options(
                store_trace = true,
                show_trace = true,
                show_every = 1,
                iterations = 8_000,
            ),
        )
        stop!()
        JLD2.save_object(snakemake.output[1], result)
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
