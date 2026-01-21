using DealerMnA
import .Demand: DemandModel, FixedCoefType, glorot_initialization
using StatsBase: cov
using LinearAlgebra
LinearAlgebra.BLAS.set_num_threads(1)
using Optim
import Random: shuffle!, seed!

seed!(78704)

if isempty(ARGS)
    snakemake!()
else
    snakemake!(ARGS[1])
end

using MPI
MPI.Init()

model = DemandModel{FixedCoefType}(; JLD2.load_object(snakemake.input["model"])...)

comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
syze = MPI.Comm_size(comm)

MPI.Barrier(comm)

struct StopSignal end

stop!() = for i in 1:(syze - 1)
        MPI.send(StopSignal(), i, 0, comm)
    end

with_logging() do
    if rank == 0
        @info "Starting optimization with $(syze) processes"
        θ = glorot_initialization(model)
        W = let
            ξ, z_j = Demand.ξ̂(model, θ)
            moments = reshape(ξ .* z_j, prod(Base.size(ξ)[1:3]), :)
            inv(cov(moments))
        end
        zips = Registrations.zips().zip_code
        result = optimize(
            θ,
            Adam(),
            Optim.Options(
                store_trace = true,
                show_trace = true,
                show_every = 1,
                iterations = 50_000,
            ),
        ) do params
            shuffle!(zips)
            for i in 1:(syze - 1)
                MPI.send(params, i, 0, comm)
                MPI.send(zips[i], i, 0, comm)
            end
            objective = Demand.Π̂(model, params, W)
            for i in 1:(syze - 1)
                signal, _ = MPI.recv(i, 0, comm)
                if signal isa StopSignal
                    @info "Process $i signaled to stop"
                    stop!()
                    error("Stopping optimization as requested by process $i")
                else
                    objective -= signal
                end
            end
            return objective
        end
        stop!()
        JLD2.save_object(snakemake.output[1], result)
    else
        while true
            params, _ = MPI.recv(0, 0, comm)
            if params isa StopSignal
                @info "Received stop signal from master"
                break
            end
            z, _ = MPI.recv(0, 0, comm)
            try
                likelihood = Demand.loglikelihood(model, params, z)
                MPI.send(likelihood, 0, 0, comm)
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
