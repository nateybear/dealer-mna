# Parallel (non-MPI) demand-estimation driver using the Dense submodule.
#
# Architecture: single Julia process; per-block dense matrices stacked across all zips;
# BLAS GEMMs + per-column softmax replace the SparseArrayKit-backed per-zip loops and
# MPI scatter/gather of scripts/demand_estimation.jl.
#
# Threading: Julia threads at the outer (per-time-block) level, BLAS threads for GEMM.
# Recommended:
#   julia --project=. --threads=<N_cores/2> scripts/demand_estimation_dense.jl
# and let BLAS use the remaining cores via the default. Tune by setting
# `BLAS.set_num_threads(...)` below if oversubscription shows up in profiling.
#
# Same parameter ordering, same inner-Newton + Woodbury block-diagonal Hessian, same
# outer NewtonTrustRegion + BHHH structure as scripts/demand_estimation.jl. Safety
# guards (warm-start guard, Float32 underflow penalty, stuck-iter detection, η bounds
# pre-screen, dϕ0 ≥ 0 fallback) are mirrored.

using DealerMnA
import .Demand: DemandModel, FixedCoefType, initial_parameters
import .Demand.Dense:
    DenseModel, DenseWork, forward_pass!, eval_ll!, eval_ll_δgrad!,
    eval_combined!, woodbury_newton_step!, param_indices
using LinearAlgebra
using Optim
import Random: seed!
import LineSearches: HagerZhang, InitialStatic
import Optim.NLSolversBase: only_fgh!, only_fg!

seed!(78704)
snakemake!()

model = DemandModel{FixedCoefType}(; JLD2.load_object(snakemake.input["model"])...)

const N_θ = length(initial_parameters(model))
const IDX = param_indices(model)
const N_δ = length(IDX.i_δ)
const N_NONδ = N_θ - N_δ
const I_NONδ_ALL = collect(IDX.i_Πᵦ[1]:IDX.i_η[end])

# Per-block GEMMs have inner dim K+1=5, so BLAS internal threading doesn't help much.
# Julia threads at the block level give the win; set BLAS=1 to avoid oversubscription.
# (The one large GEMM/SYRK in eval_combined! — the 22×22 BHHH and Π_grad — is tiny.)
LinearAlgebra.BLAS.set_num_threads(1)

with_logging() do
    @info "Building DenseModel" Z_zips = length(Registrations.zips().zip_code) N = N_δ
    zips = Registrations.zips().zip_code
    dm = DenseModel(model, zips)
    work = DenseWork(dm)
    @info "DenseModel built" n_blocks = dm.n_blocks

    θ =
        if haskey(snakemake.input, "params_restart") &&
            !isempty(snakemake.input["params_restart"]) &&
            isfile(snakemake.input["params_restart"])
            JLD2.load_object(snakemake.input["params_restart"]).minimizer
        else
            initial_parameters(model)
        end
    θ = Float32.(θ)
    i_δ = collect(IDX.i_δ)
    i_nonδ = collect(I_NONδ_ALL)
    π₀ = Float32.(θ[i_nonδ])

    # Per-call buffers (allocated once, reused)
    grad_buf = zeros(Float32, N_θ)
    bhhh_buf = zeros(Float32, N_NONδ, N_NONδ)
    step_buf = zeros(Float32, N_δ)
    δ_grad_buf = zeros(Float32, N_δ)

    # Outer-iter state
    _cached_π = fill(NaN32, length(i_nonδ))
    _cached_θ_opt = copy(θ)
    _inner_iter = Ref(0)
    _last_ll = Ref(NaN32)             # NLL at most recently accepted outer point
    _cached_conv_ll = Ref(NaN32)      # NLL at δ*(π) on inner Newton convergence
    _cached_conv_grad = zeros(Float32, length(i_nonδ))  # ∇NLL wrt π
    _cached_conv_bhhh = 1.0f-4 * Matrix{Float32}(I, length(i_nonδ), length(i_nonδ))

    ls = HagerZhang()

    nonδ_labels =
        let D = string.(Demand.DEMOGRAPHIC_ATTRIBUTES), P = string.(Demand.PRODUCT_ATTRIBUTES)
            vcat(
                ["Π[$d,$p]" for p in P for d in D],
                ["Πp[$d]" for d in D],
                ["η_baseline", "η_dist"],
            )
        end

    # Bounds pre-screen — identical to MPI script. Cheap rejection of obviously bad proposals.
    function bounds_violated(π)
        η_dist = π[end]
        η_baseline = π[end - 1]
        return abs(η_dist) > 5 || abs(η_baseline) > 10 ||
               any(>(15), abs.(π[1:(end - 2)]))
    end

    function concentrated_θ!(π)
        if π == _cached_π
            return _cached_θ_opt
        end
        _cached_conv_ll[] = NaN32
        _inner_iter[] += 1
        id = _inner_iter[]
        @info "Starting inner optimization #$id"

        θ_tmp = copy(θ)
        θ_tmp[i_nonδ] .= π
        # Warm start: reuse last δ*
        θ_tmp[i_δ] .= _cached_θ_opt[i_δ]

        # Warm start guard
        if !isnan(_last_ll[]) && _last_ll[] > 0
            ϕ_warm = -eval_ll!(work, dm, θ_tmp)
            if ϕ_warm > 1.0f3 * _last_ll[]
                @info "  inner #$id: warm start NLL=$(round(ϕ_warm; sigdigits=4)) >> prev $(round(_last_ll[]; sigdigits=4)), resetting δ"
                θ_tmp[i_δ] .= θ[i_δ]
            end
        end

        δ_at_inner_start = copy(θ_tmp[i_δ])

        newton_iter = 0
        gnorm = Inf32
        stuck_iters = 0
        unstuck_iters = 0
        trying_to_unstick = false
        bad_descent_count = 0

        while true
            newton_iter += 1
            if newton_iter > 30
                @warn "  inner #$id: exceeded 30 newton iters (gnorm=$gnorm)"
                break
            end

            t_combined = @elapsed begin
                ll = eval_combined!(work, dm, θ_tmp; grad = grad_buf, bhhh = bhhh_buf)
            end
            ϕ0 = -ll                          # NLL
            g_δ_nll = -view(grad_buf, i_δ)    # ∂NLL/∂δ
            gnorm = Float32(norm(g_δ_nll))

            # Float32 CCP underflow guard — try uniform δ shift, then bail if it fails
            if isapprox(ϕ0, 0)
                @warn "  inner #$id iter $newton_iter: NLL≈0, trying +500 δ shift"
                θ_tmp[i_δ] .+= 500.0f0
                ll = eval_ll!(work, dm, θ_tmp)
                if isapprox(-ll, 0)
                    @warn "  inner #$id: δ shift didn't help, breaking"
                    break
                end
                continue
            end

            if gnorm ≤ 1.0f-2
                @info "  inner #$id converged in $newton_iter newton iters (gnorm=$gnorm)"
                _cached_conv_ll[] = ϕ0
                _cached_conv_grad .= .-grad_buf[i_nonδ]
                if tr(bhhh_buf) > 1e5
                    fill!(_cached_conv_bhhh, 0.0f0)
                    @inbounds for i in axes(_cached_conv_bhhh, 1)
                        _cached_conv_bhhh[i, i] = bhhh_buf[i, i] + 1.0f-4
                    end
                else
                    _cached_conv_bhhh .= bhhh_buf
                end
                break
            end

            # Woodbury Newton step over δ. _woodbury uses ∂ℓ/∂δ (ascent gradient) →
            # ascent direction for ℓ = descent direction for NLL.
            δ_grad_ascent = view(grad_buf, i_δ)  # +∂ℓ/∂δ
            woodbury_newton_step!(step_buf, work, dm, δ_grad_ascent)
            d = step_buf

            # dϕ0 = ∇NLL · d. Must be < 0 for HagerZhang. Fall back to -g/gnorm otherwise.
            dϕ0 = Float32(dot(g_δ_nll, d))
            if !(dϕ0 < 0) || (stuck_iters ≥ 3) ||
                (trying_to_unstick && unstuck_iters ≤ 1)
                d .= -g_δ_nll ./ max(gnorm, 1.0f-8)
                dϕ0 = Float32(dot(g_δ_nll, d))
                bad_descent_count = !(dϕ0 < 0) ? bad_descent_count + 1 : 0
                if bad_descent_count ≥ 3
                    @warn "  inner #$id iter $newton_iter: 3× non-descent in a row, breaking"
                    break
                end
                trying_to_unstick = true
                stuck_iters = 0
            else
                bad_descent_count = 0
            end

            @info "  inner #$id newton iter $newton_iter: " *
                "gnorm=$(round(gnorm; sigdigits=4)) dϕ0=$(round(dϕ0; sigdigits=4)) " *
                "combined_time=$(round(t_combined; sigdigits=3))s"

            δ_cur = copy(θ_tmp[i_δ])
            θ_trial = copy(θ_tmp)

            function ϕ(α)
                θ_trial[i_δ] .= δ_cur .+ Float32(α) .* d
                return -eval_ll!(work, dm, θ_trial)
            end
            function dϕ(α)
                θ_trial[i_δ] .= δ_cur .+ Float32(α) .* d
                eval_ll_δgrad!(work, dm, θ_trial; δ_grad = δ_grad_buf)
                return -Float32(dot(δ_grad_buf, d))
            end
            # Fuse ϕ and dϕ — one forward pass gives both ll and δ-gradient.
            function ϕdϕ(α)
                θ_trial[i_δ] .= δ_cur .+ Float32(α) .* d
                ll = eval_ll_δgrad!(work, dm, θ_trial; δ_grad = δ_grad_buf)
                return (-ll, -Float32(dot(δ_grad_buf, d)))
            end

            local alpha::Float32, phi_alpha::Float32
            try
                a, p = ls(ϕ, dϕ, ϕdϕ, 1.0f0, ϕ0, dϕ0)
                alpha = Float32(a)
                phi_alpha = Float32(p)
            catch e
                @warn "  inner #$id iter $newton_iter: line search threw ($e), breaking"
                break
            end

            if alpha < 1.0f-5
                stuck_iters += 1
                unstuck_iters = 0
            else
                if trying_to_unstick
                    unstuck_iters += 1
                    unstuck_iters ≥ 2 && (trying_to_unstick = false)
                end
                stuck_iters = 0
            end

            @info "  inner #$id newton iter $newton_iter: NLL=$(round(phi_alpha; sigdigits=6)) α=$(round(alpha; sigdigits=4))"
            θ_tmp[i_δ] .+= alpha .* d
        end

        # Unconverged exit penalty: restore δ if gnorm still huge
        if gnorm > 1.0f4
            @warn "  inner #$id: exiting with gnorm=$gnorm > 1e4, restoring start δ"
            θ_tmp[i_δ] .= δ_at_inner_start
            # _cached_conv_ll remains NaN — outer_f detects and returns penalty
        end

        _cached_π .= π
        _cached_θ_opt .= θ_tmp
        θ .= θ_tmp
        return _cached_θ_opt
    end

    function outer_optimize(π₀)
        _prev_π = copy(π₀)
        function outer_fgh!(F, G, H, π)
            if bounds_violated(π)
                isnothing(G) || (G .= 0)
                isnothing(H) || (H .= 0)
                return isnothing(F) ? nothing : 1.0f10
            end
            Δπ = π .- _prev_π
            step_lines = join(
                [
                    "  " *
                    rpad(nonδ_labels[k], 36) *
                    "val=$(rpad(string(round(Float64(π[k]); sigdigits=4)), 14))" *
                    "Δ=$(round(Float64(Δπ[k]); sigdigits=4))" for k in eachindex(nonδ_labels)
                ],
                "\n",
            )
            @info "\nstep1 outer_f: step_norm=$(round(Float64(norm(Δπ)); sigdigits=4))\n" *
                step_lines
            _prev_π .= π
            concentrated_θ!(π)
            ll = _cached_conv_ll[]
            if isnan(ll) || isapprox(ll, 0)
                isnothing(G) || (G .= 0)
                isnothing(G) || (H .= 0)
                return isnothing(F) ? nothing : 1.0f10
            end
            _last_ll[] = ll
            isnothing(G) || (G .= _cached_conv_grad)
            isnothing(H) || (H .= _cached_conv_bhhh)
            return isnothing(F) ? nothing : ll
        end

        _checkpoint_path = snakemake.output[1] * ".checkpoint.jld2"
        _cb_iter = Ref(0)
        function checkpoint_cb(state)
            _cb_iter[] += 1
            if _cb_iter[] % 25 == 0
                JLD2.save_object(
                    _checkpoint_path, (minimizer = _cached_θ_opt[i_nonδ], value = state.f_x)
                )
                @info "  checkpoint saved at outer_iter=$(_cb_iter[]) NLL=$(round(state.f_x; sigdigits=6))"
            end
            return false
        end

        return optimize(
            only_fgh!(outer_fgh!),
            π₀,
            NewtonTrustRegion(),
            Optim.Options(
                store_trace = true,
                extended_trace = true,
                show_trace = true,
                iterations = 5000,
                g_abstol = 1.0f-3,
                x_abstol = NaN,
                x_reltol = NaN,
                f_abstol = NaN,
                f_reltol = NaN,
                allow_f_increases = true,
                callback = checkpoint_cb,
            ),
        )
    end

    @info "Starting outer optimization"
    result = outer_optimize(π₀)
    @info "\nEstimation done: NLL=$(round(result.minimum; sigdigits=6)) converged=$(Optim.converged(result))\n"

    JLD2.save_object(snakemake.output[1], result)
end
