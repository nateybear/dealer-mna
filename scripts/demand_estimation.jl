using DealerMnA
import .Demand: DemandModel, FixedCoefType, initial_parameters
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
    out = parent_handle(T, model, W, zips[syze], params)
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
    normalize!(out, T)
    return out
end

function parent_handle(::Type{LikelihoodRequest}, model, W, z, params)
    islikelihood = snakemake.params["objective"] == "likelihood"
    if islikelihood
        -Demand.loglikelihood(model, params, z)
    else
        Demand.Π̂(model, params, W)
    end
end

function parent_handle(::Type{GradientRequest}, model, W, z, params)
    islikelihood = snakemake.params["objective"] == "likelihood"
    if islikelihood
        -Demand.dloglikelihood(model, params, z)
    else
        Demand.dΠ̂(model, params, W)
    end
end

child_handle(r::LikelihoodRequest) = Demand.loglikelihood(model, r.params, r.z)
child_handle(r::GradientRequest) = Demand.dloglikelihood(model, r.params, r.z)

normalize!(out, ::Type{LikelihoodRequest}) = nothing
function normalize!(out, ::Type{GradientRequest})
    i = Demand.@indexof delta out model
    normalize!(@view out[i])
    j = last(i) + 1
    normalize!(@view out[j:end]) 
    return nothing
end

function normalize!(out)
    g = norm(out) 
    if g > 1
        out ./= g
    end
end

stop!() =
    for i in 1:(syze - 1)
        MPI.send(StopSignal(), i, 0, comm)
    end

with_logging() do
    if rank == 0
        @info "Starting optimization with $(syze) processes"
        θ = if haskey(snakemake.input, "params_restart") && 
                !isempty(snakemake.input["params_restart"]) &&
                isfile(snakemake.input["params_restart"])
            JLD2.load_object(snakemake.input["params_restart"]).minimizer
        else
            initial_parameters(model)
        end
        zips = Registrations.zips().zip_code
        i = 0
        lr_max = 5f-5
        lr_min = 1f-6
        T = 16 
        while i < T
            W = Demand.optimalW(model, θ)
            f = OnceDifferentiable(
                parent_scatter$(LikelihoodRequest, zips, W),
                parent_scatter$(GradientRequest, zips, W),
                θ,
                inplace = false,
            )
            i += 1
            lr = lr_min + 0.5 * (lr_max - lr_min) * (1 + cos(i * pi / T))
            @info "In loop $i"
            result = optimize(
                f,
                θ,
                Adam(alpha = Float32(lr), beta_mean = 0.9f0, beta_var = 0.99f0, epsilon=1f-8),
                Optim.Options(
                    store_trace = true,
                    show_trace = true,
                    show_every = 1,
                    iterations = 500,
                ),
            )    
            JLD2.save_object(snakemake.output[1], result)
            p, ext = splitext(snakemake.output[1])
            obj = snakemake.params["objective"]
            JLD2.save_object("$(p)_$(obj)_$(i)$(ext)", result)
            θ = result.minimizer
        end
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
