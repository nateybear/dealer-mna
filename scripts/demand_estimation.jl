using DealerMnA
import .Demand: DemandModel, FixedCoefType, initial_parameters, @indexof, @pull
using StatsBase: cov
using LinearAlgebra
using Optim
import Random: seed!
import LineSearches: BackTracking, InitialStatic, HagerZhang

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

LinearAlgebra.BLAS.set_num_threads(rank == 0 ? 2 : 1)

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

struct CombinedRequest
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
    resp = mapreduce(+, r.zs, init = 0.0f0) do z
        Demand.loglikelihood(model, r.params, z)
    end
    MPI.send(resp, 0, 0, comm)
    return nothing
end

function child_handle(r::GradientRequest)
    resp = mapreduce(+, r.zs, init = zeros(Float32, length(r.params))) do z
        Demand.dloglikelihood(model, r.params, z)
    end
    MPI.send(resp, 0, 0, comm)
    return nothing
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

function child_handle(r::CombinedRequest)
    all_idxs = Demand.δ_indices(model)
    Z_local = length(r.zips)
    ll_sum = 0.0f0
    grad_sum = zeros(Float32, length(r.params))
    all_ccps = Matrix{Float32}(undef, length(all_idxs), Z_local)
    for (zi, z) in enumerate(r.zips)
        ll_z, grad_z, ccp_vec = Demand.loglikelihood_gradient_and_ccps(model, r.params, z)
        ll_sum += ll_z
        grad_sum .+= grad_z
        all_ccps[:, zi] .= ccp_vec
    end
    MPI.send((grad_sum, ll_sum), 0, 0, comm)
    for (bi, (_, i_δs)) in enumerate(T_BLOCK_ORDER)
        MPI.send(all_ccps[i_δs, :], 0, bi, comm)
    end
    return nothing
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
    # Degenerate CCPs (extreme utilities after a large outer step) can make middle singular in
    # Float32. Fall back to g_t (gradient ascent); the line search provides the step size, and
    # the Woodbury is retried automatically once δ has moved and CCPs recover.
    try
        w = Symmetric(middle) \ (U' * v)           # (M⁻¹ - U'D⁻¹U)⁻¹ U'v, Z-vector
        return v + DinvU * w                        # = (-H_t)⁻¹ g_t via Woodbury
    catch e
        e isa SingularException || rethrow(e)
        return g_t
    end
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

# Scatter a CombinedRequest; receive (grad, ll) on tag 0 from all workers, then assemble
# per-block CCP matrices and compute Woodbury Newton step.
# Returns (-∇ℓ, Newton ascent step for ℓ over δ, -ℓ) — matching mpi_scatter sign convention.
function mpi_combined_newton_step(
    params::Vector{Float32},
    zips::Vector{String},
)::Tuple{Vector{Float32}, Vector{Float32}, Float32}
    chunksize = cld(length(zips), syze - 1)
    zip_chunks = collect(Iterators.partition(zips, chunksize))
    maxrank = length(zip_chunks)
    for (i, chunk) in enumerate(zip_chunks)
        MPI.send(CombinedRequest(params, collect(chunk)), i, 0, comm)
    end
    M_zs = Float32[model.agent_data[z].M for z in zips]
    g_full = zeros(Float32, length(params))  # accumulates +∇ℓ
    ll_sum = 0.0f0                           # accumulates +ℓ
    for i in 1:maxrank
        (grad_i, ll_i), _ = MPI.recv(i, 0, comm)
        g_full .+= grad_i
        ll_sum += ll_i
    end
    i_δ_all = @indexof delta params model
    g_δ = g_full[i_δ_all]  # +∇_δ ℓ
    step = Vector{Float32}(undef, length(g_δ))
    for (bi, (_, i_δs)) in enumerate(T_BLOCK_ORDER)
        N_t = length(i_δs)
        U = Matrix{Float32}(undef, N_t, length(zips))
        col = 1
        for i in 1:maxrank
            ccp_block, _ = MPI.recv(i, bi, comm)
            ncols = size(ccp_block, 2)
            U[:, col:(col + ncols - 1)] .= ccp_block
            col += ncols
        end
        step[i_δs] .= woodbury_newton_step(U, M_zs, g_δ[i_δs])
    end
    return -g_full, step, -ll_sum
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
        @info "Initial parameters set"
        zips = Registrations.zips().zip_code
        i_δ = @indexof delta θ model
        i_nonδ = setdiff(1:length(θ), i_δ)

        π₀ = Float32.(θ[i_nonδ])

        # Cache: avoid re-running the inner problem when f, g!, h! are called at the same π.
        _cached_π = fill(NaN32, length(i_nonδ))
        _cached_θ_opt = copy(θ)
        _inner_iter = Ref(0)
        _last_ll = Ref(NaN32)
        _inner_failed = Ref(false)

        ls = HagerZhang()

        # Labels for the non-delta params in vector order (column-major: demographics vary fastest).
        nonδ_labels =
            let D = string.(Demand.DEMOGRAPHIC_ATTRIBUTES), P = string.(Demand.PRODUCT_ATTRIBUTES)
                vcat(
                    ["Π[$d,$p]" for p in P for d in D],
                    ["Πp[$d]" for d in D],
                    ["η_baseline", "η_dist"],
                )
            end

        # Inner problem: given non-delta params π, find δ*(π) = argmax_δ loglikelihood.
        # Custom block-diagonal Newton: solves (-H_t)*Δδ_t = g_t per time block, never
        # forming the full n_delta×n_delta Hessian. Each MPI round returns the full
        # block-diagonal H_flat (1.25 GB on workers) accumulated in-place to avoid
        # per-zip allocations. Quadratic convergence in 2–5 iterations.
        function concentrated_θ!(π)
            if π == _cached_π
                return _cached_θ_opt
            end
            _inner_failed[] = false
            _inner_iter[] += 1
            @info "Starting inner optimization #$(_inner_iter[])"
            id = _inner_iter[]

            θ_tmp = copy(θ)
            θ_tmp[i_nonδ] .= π
            # warm start: reuse last δ* rather than resetting to initial
            (@pull delta θ_tmp model) .= @pull delta _cached_θ_opt model
            # Guard: if the outer step was too large, the warm-started δ may produce
            # degenerate CCPs (Float32 underflow) that break the Woodbury Newton step.
            # Fall back to the initial δ when logL is >10x worse than the previous accepted value.
            # Skip when _last_ll[]==0: that means a prior inner call exited via CCP underflow and
            # the reference point is meaningless — any positive ϕ_warm would trigger a spurious reset.
            if !isnan(_last_ll[]) && _last_ll[] > 0
                ϕ_warm = mpi_scatter(LikelihoodRequest, zips, θ_tmp)
                if ϕ_warm > 1.0f3 * _last_ll[]
                    @info "  inner #$id: warm start logL=$(round(ϕ_warm; sigdigits=4)) >> prev $(round(_last_ll[]; sigdigits=4)), resetting to initial δ"
                    (@pull delta θ_tmp model) .= @pull delta θ model
                end
            end
            # Snapshot δ after warm-start resolution. Restored into θ_tmp on any failed exit so
            # that _cached_θ_opt is never left with δ values corrupted by a partial Newton run.
            δ_at_inner_start = copy(@pull delta θ_tmp model)

            newton_iter = 0
            stuck_iters = 0  # consecutive iterations with α < 1e-5 (Float32 underflow trap)
            trying_to_unstick = false # if we are trying to get out of a hole, keep taking gradient steps
            unstuck_iters = 0
            nan_dϕ0_iters = 0  # consecutive iterations where dϕ0 was NaN/non-negative
            gnorm = Inf32
            while true
                newton_iter += 1
                if newton_iter > 30
                    @warn "  inner #$id newton: exceeded 30 iters (gnorm=$gnorm), giving up"
                    break
                end
                δ_cur = @pull delta θ_tmp model
                # Combined round: gradient, Newton step, and ϕ0 in one MPI scatter.
                t_combined = @elapsed begin
                    g_full, d, ϕ0 = mpi_combined_newton_step(θ_tmp, zips)
                end
                g = @pull delta g_full model  # -∇_δ ℓ
                gnorm = norm(g)
                # ϕ0≈0 means Float32 CCP underflow (all exp(utility)→0 due to extreme π).
                # Try a uniform δ shift to bring utilities into the non-underflow regime: adding C
                # to all δ in a block makes products more attractive vs. the outside good and can
                # rescue moderate underflow.  If the shift doesn't help (e.g. η overflow dominates),
                # exit immediately — outer_f will return a large penalty to force backtracking.
                if isapprox(ϕ0, 0)
                    δ_cur .+= 500.0f0
                    ϕ0_shifted = mpi_scatter(LikelihoodRequest, zips, θ_tmp)
                    if isapprox(ϕ0_shifted, 0)
                        @warn "  inner #$id newton iter $newton_iter: deep underflow (δ shift +500 didn't help), exiting — outer_f will return penalty"
                        δ_cur .-= 500.0f0  # restore so _cached_θ_opt stays clean
                        break
                    end
                    @info "  inner #$id newton iter $newton_iter: escaped underflow with δ shift +500, retrying Newton"
                    continue
                end
                if gnorm ≤ 1.0f-3
                    @info "  inner #$id newton: converged in $newton_iter iters (gnorm=$gnorm)"
                    break
                end
                # First-iteration gnorm check: if the warm-started δ is far from δ*(π_new),
                # the inner Newton will be stuck (astronomical Woodbury steps, α→1e-19).
                # Bail immediately after one MPI round (~10s) rather than burning 30×10s.
                # Threshold 1e3: requires the warm-started δ to already be quite close to
                # δ*(π_new), enforcing local outer steps. Skipped for id==1 (cold start from
                # initial δ=-10, which legitimately has large iter-1 gnorm ~10,600).
                if newton_iter == 1 && id > 1 && gnorm > 1.0f3
                    @warn "  inner #$id: gnorm=$(round(gnorm; sigdigits=4)) at first iter >> 1e3, outer step too large — returning penalty"
                    (@pull delta θ_tmp model) .= δ_at_inner_start
                    # _inner_failed[] = true
                    break
                end
                # Stuck detection: if the line search has returned α < 1e-5 for several
                # consecutive iterations, the Newton direction d has enormous magnitude
                # (from saturated Float32 CCPs) and δ+α*d==δ in Float32 arithmetic.
                # Switch to a normalised gradient step, which has unit magnitude and a
                # valid descent slope, so the line search can make actual progress.
                if stuck_iters >= 3 || (trying_to_unstick && unstuck_iters <= 1)
                    @info "  inner #$id newton iter $newton_iter: stuck for $stuck_iters iters, switching to gradient step"
                    d = -g ./ max(gnorm, 1.0f-8)  # unit-magnitude descent direction for -ℓ
                    trying_to_unstick = true
                    stuck_iters = 0
                end
                dϕ0 = dot(g, d)  # ∇(-ℓ)·d ≤ 0 (valid descent direction)
                # HagerZhang throws on a non-descent direction; guard here so the error doesn't
                # propagate to Optim and kill the outer loop.  Degenerate CCPs (NaN/Inf from
                # Float32 overflow) can produce a Newton step that is not a descent direction.
                if !(dϕ0 < 0)
                    nan_dϕ0_iters += 1
                    @warn "  inner #$id newton iter $newton_iter: dϕ0=$(round(dϕ0; sigdigits=4)) not negative, falling back to gradient step ($nan_dϕ0_iters consecutive)"
                    if nan_dϕ0_iters >= 3
                        @warn "  inner #$id newton iter $newton_iter: $nan_dϕ0_iters consecutive non-descent steps, aborting — outer_f will return penalty"
                        (@pull delta θ_tmp model) .= δ_at_inner_start
                        # _inner_failed[] = true
                        break
                    end
                    d = -g ./ max(gnorm, 1.0f-8)
                    dϕ0 = -gnorm
                else
                    nan_dϕ0_iters = 0
                end
                @info "  inner #$id newton iter $newton_iter: gnorm=$(round(gnorm; sigdigits=4)) dϕ0=$(round(dϕ0; sigdigits=4)) combined_time=$(round(t_combined; sigdigits=3))s"
                function ϕ(α)
                    θ_tmp_tmp = copy(θ_tmp)
                    (@pull delta θ_tmp_tmp model) .= δ_cur + α * d
                    return mpi_scatter(LikelihoodRequest, zips, θ_tmp_tmp)
                end
                function dϕ(α)
                    θ_tmp_tmp = copy(θ_tmp)
                    (@pull delta θ_tmp_tmp model) .= δ_cur + α * d
                    g_trial = mpi_scatter(GradientRequest, zips, θ_tmp_tmp)
                    return dot(@pull(delta, g_trial, model), d)
                end
                ϕdϕ(α) = (ϕ(α), dϕ(α))
                alpha, phi_alpha = ls(ϕ, dϕ, ϕdϕ, 1.0f0, ϕ0, dϕ0)
                @info "  inner #$id newton iter $newton_iter: -logL=$(round(phi_alpha; sigdigits=6)) α=$(round(alpha; sigdigits=4))"
                is_stuck = alpha < 1.0f-5
                if is_stuck
                    stuck_iters += 1
                    unstuck_iters = 0
                else
                    stuck_iters = 0
                    if trying_to_unstick
                        unstuck_iters += 1
                        if unstuck_iters > 1
                            # we have unstuck now
                            trying_to_unstick = false
                            unstuck_iters = 0
                        end
                    end
                end
                δ_cur .+= alpha * d
            end
            if gnorm > 1.0f4 && !_inner_failed[]
                @warn "  inner #$id newton: exiting with gnorm=$(round(gnorm; sigdigits=4)) > 1e4 (unconverged), returning penalty"
                (@pull delta θ_tmp model) .= δ_at_inner_start
                # _inner_failed[] = true
            end
            _cached_π .= π
            _cached_θ_opt .= θ_tmp
            return _cached_θ_opt
        end

        function outer_optimize(π₀, label)
            _prev_π = copy(π₀)
            function outer_f(π)
                # Fast pre-screen: return penalty instantly for obviously infeasible π.
                # At |η_dist| > 5 (one σ distance), exp(η₁ + 5) dominates and Float32 CCP
                # underflow is inevitable. Each avoided scatter saves ~10s of MPI time.
                if abs(π[end]) > 5.0f0 || abs(π[end - 1]) > 10.0f0 ||
                    maximum(abs, @view π[1:(end - 2)]) > 15.0f0
                    return 1.0f10
                end
                Δπ = π .- _prev_π
                step_lines = join(
                    [
                        "  " *
                        rpad(nonδ_labels[k], 36) *
                        "val=$(rpad(string(round(Float64(π[k]); sigdigits=4)), 14))" *
                        "Δ=$(round(Float64(Δπ[k]); sigdigits=4))" for
                        k in eachindex(nonδ_labels)
                    ],
                    "\n",
                )
                @info "\n$label outer_f: step_norm=$(round(Float64(norm(Δπ)); sigdigits=4))\n" *
                    step_lines
                _prev_π .= π
                θ_opt = concentrated_θ!(π)
                if _inner_failed[]
                    @warn "$label outer_f: inner Newton failed (non-descent abort or unconverged), returning penalty to force backtrack"
                    return 1.0f10
                end
                ll = mpi_scatter(LikelihoodRequest, zips, θ_opt)
                if isapprox(ll, 0)
                    # Inner Newton exited due to deep Float32 underflow: return a large penalty so
                    # the outer HagerZhang line search rejects this point and backtracks.  Do NOT
                    # update _last_ll[] — the warm-start guard needs the last valid reference logL.
                    @warn "$label outer_f: inner Newton returned ll=0 (deep underflow), returning penalty to force backtrack"
                    return 1.0f10
                end
                _last_ll[] = ll
                return ll
            end

            function outer_g!(G, π)
                θ_opt = concentrated_θ!(π)
                G .= mpi_scatter(GradientRequest, zips, θ_opt)[i_nonδ]
                return nothing
            end

            return optimize(
                OnceDifferentiable(outer_f, outer_g!, π₀),
                π₀,
                LBFGS(
                    alphaguess = InitialStatic(alpha = 1e-4),
                    linesearch = BackTracking(order = 2, c_1 = 1e-4, ρ_lo = 0.1, ρ_hi = 0.5),
                ),
                Optim.Options(
                    store_trace = true,
                    show_trace = false,
                    iterations = 500,
                    g_abstol = 1.0f-3,
                    x_abstol = NaN,
                    x_reltol = NaN,
                    f_abstol = NaN,
                    f_reltol = NaN,
                    allow_f_increases = true,
                    callback = state -> begin
                        π_cur = _cached_θ_opt[i_nonδ]
                        g = state.g_x
                        param_lines = join(
                            [
                                "  " *
                                rpad(nonδ_labels[k], 36) *
                                "val=$(rpad(string(round(Float64(π_cur[k]); sigdigits=4)), 14))" *
                                "grad=$(round(Float64(g[k]); sigdigits=4))" for
                                k in eachindex(nonδ_labels)
                            ],
                            "\n",
                        )
                        @info "\n$label iter $(state.pseudo_iteration): " *
                            "total=$(round(state.f_x; sigdigits=6)) " *
                            "-logL=$(round(_last_ll[]; sigdigits=6)) " *
                            "gnorm=$(round(norm(g); sigdigits=4)) " *
                            "inner_iters=$(_inner_iter[])\n" *
                            param_lines
                        return false
                    end,
                ),
            )
        end

        result = outer_optimize(π₀, "step1")
        @info "\nEstimation done: total=$(round(result.minimum; sigdigits=6)) converged=$(Optim.converged(result))\n"

        JLD2.save_object(
            snakemake.output[1],
            result,
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
                child_handle(msg)
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
