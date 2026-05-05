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
        W = Demand.optimalW(model, θ)
        i_δ = @indexof delta θ model
        _last_ll = Ref(NaN32)
        _last_gmm = Ref(NaN32)

        # Warm start: find a good δ at the initial non-delta params before joint optimization.
        # Only runs when starting from scratch (initial_parameters sets δ = -3 uniformly).
        if !haskey(snakemake.input, "params_restart") ||
            isempty(snakemake.input["params_restart"]) ||
            !isfile(snakemake.input["params_restart"])
            @info "Warm-starting delta via likelihood maximization ($(length(i_δ)) params)"
            δ₀ = copy(@pull delta θ model)
            warm_result = optimize(
                OnceDifferentiable(
                    function (δ_w)
                        θ_tmp = copy(θ)
                        (@pull delta θ_tmp model) .= δ_w
                        return mpi_scatter(LikelihoodRequest, zips, θ_tmp)
                    end,
                    function (δ_w)
                        θ_tmp = copy(θ)
                        (@pull delta θ_tmp model) .= δ_w
                        return mpi_scatter(GradientRequest, zips, θ_tmp)[i_δ]
                    end,
                    δ₀,
                    inplace = false,
                ),
                δ₀,
                LBFGS(m = 10, linesearch = LineSearches.BackTracking()),
                Optim.Options(
                    iterations = 200,
                    g_tol = 1.0f-3,
                    allow_f_increases = true,
                    callback = state -> begin
                        @info "warm-start iter $(state.pseudo_iteration): " *
                            "-logL=$(round(state.f_x; sigdigits=6)) "
                        return state.f_x ≤ 0.0f0
                    end,
                ),
            )
            (@pull delta θ model) .= warm_result.minimizer
            @info "Warm start done: -logL=$(round(warm_result.minimum; sigdigits=6))"
        end

        lbfgs_opts = LBFGS(m = 20, linesearch = LineSearches.BackTracking())

        function joint_optimize(θ₀, W_cur, label)
            return optimize(
                OnceDifferentiable(
                    function (θ_cur)
                        ll = mpi_scatter(LikelihoodRequest, zips, θ_cur)
                        gmm = Demand.Π̂(model, θ_cur, W_cur)
                        _last_ll[] = ll
                        _last_gmm[] = gmm
                        return ll + gmm
                    end,
                    θ_cur ->
                        mpi_scatter(GradientRequest, zips, θ_cur) .+
                        Demand.dΠ̂(model, θ_cur, W_cur),
                    θ₀,
                    inplace = false,
                ),
                θ₀,
                lbfgs_opts,
                Optim.Options(
                    store_trace = true,
                    show_trace = true,
                    iterations = 5000,
                    f_tol = 1.0f-9,
                    g_tol = 1.0f-6,
                    allow_f_increases = true,
                ),
            )
        end

        # Step 1: optimize with initial W.
        result1 = joint_optimize(θ, W, "step1")
        θ̂₁ = result1.minimizer
        @info "Step 1 done: total=$(round(result1.minimum; sigdigits=6)) converged=$(Optim.converged(result1))"

        # Step 2: recompute optimal W at θ̂₁, re-optimize for efficiency.
        W_opt = Demand.optimalW(model, θ̂₁)
        @info "Recomputed optimal W for step 2"
        result2 = joint_optimize(θ̂₁, W_opt, "step2")
        @info "Step 2 done: total=$(round(result2.minimum; sigdigits=6)) converged=$(Optim.converged(result2))"

        JLD2.save_object(
            snakemake.output[1],
            (; minimizer = result2.minimizer, step1 = result1, step2 = result2),
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
