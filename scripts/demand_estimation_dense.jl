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
    DenseModel, DenseWork, EnvelopeHessWork,
    forward_pass!, eval_ll!, eval_ll_δgrad!,
    eval_combined!, woodbury_newton_step!,
    eval_envelope_hessian!, definiteness_shift!, param_indices
using LinearAlgebra
using Optim
import Random: seed!
import LineSearches: HagerZhang, InitialStatic, InitialHagerZhang
import Optim.NLSolversBase: only_fgh!, only_fg!
import Optim: AffineSimplexer
import JuMP
using KNITRO # will be a no-op on computers without knitro

using ArgParse

# -------------------------------------------------------------------------------------------------
# CLI parsing
# -------------------------------------------------------------------------------------------------

function build_argparse_settings()
    settings = ArgParseSettings()
    @add_arg_table settings begin
        "nelder-mead"
        action = :command
        "newton"
        action = :command
        "newton-tro"
        action = :command
        "bfgs"
        action = :command
        "knitro"
        action = :command
        "--rand-offset"
        arg_type = Float64
        default = 0.0
        "--prefix"
        default = ""
        "--fd-check"
        action = :store_true
        help = "Run the finite-difference envelope-Hessian verification and exit (no optimization)."
        "--fd-check-h"
        arg_type = Float64
        default = 1.0e-3
        help = "Step size for --fd-check central differences."
    end
    @add_arg_table settings["nelder-mead"] begin
        "-a"
        arg_type = Float64
        default = 0.5
        "-b"
        arg_type = Float64
        default = 0.5
    end
    @add_arg_table settings["newton"] begin
        "--scale"
        action = :store_true
        "--alpha"
        arg_type = Float64
        default = 0.1
    end
    @add_arg_table settings["newton-tro"] begin
        "--initial-delta"
        arg_type = Float64
        default = 0.01
    end
    @add_arg_table settings["bfgs"] begin
        "--scale"
        action = :store_true
        "--alpha"
        arg_type = Float64
        default = 0.1
    end
    @add_arg_table settings["knitro"] begin
        "--algorithm"
        arg_type = Int
        default = 0
        "--hessopt"
        arg_type = Int
        default = 1
        "--no-hessian"
        action = :store_true
        "--maxit"
        arg_type = Int
        default = 1000
        "--outlev"
        arg_type = Int
        default = 4
        "--feastol"
        arg_type = Float64
        default = 1e-6
        "--opttol"
        arg_type = Float64
        default = 1e-3
        "--xtol"
        arg_type = Float64
        default = 1e-8
        "--bar-initpt"
        arg_type = Int
        default = -1
        "--honorbnds"
        arg_type = Int
        default = 1
        "--full"
        action = :store_true
    end
    return settings
end

# -------------------------------------------------------------------------------------------------
# Misc helpers
# -------------------------------------------------------------------------------------------------

# Bounds pre-screen — identical to MPI script. Cheap rejection of obviously bad proposals.
function bounds_violated(π)
    η_dist = π[end]
    η_baseline = π[end - 1]
    return abs(η_dist) > 5 || abs(η_baseline) > 10 ||
           any(>(15), abs.(π[1:(end - 2)]))
end

function build_nonδ_labels()
    D = string.(Demand.DEMOGRAPHIC_ATTRIBUTES)
    P = string.(Demand.PRODUCT_ATTRIBUTES)
    return vcat(
        ["Π[$d,$p]" for p in P for d in D],
        ["Πp[$d]" for d in D],
        ["η_baseline", "η_dist"],
    )
end

function format_step_log(labels, π, prev_π)
    Δπ = π .- prev_π
    step_lines = join(
        [
            "  " * rpad(labels[k], 36) *
            "val=$(rpad(string(round(Float64(π[k]); sigdigits=4)), 14))" *
            "Δ=$(round(Float64(Δπ[k]); sigdigits=4))" for k in eachindex(labels)
        ],
        "\n",
    )
    return "\nstep1 outer_f: step_norm=$(round(Float64(norm(Δπ)); sigdigits=4))\n" *
           step_lines
end

function build_initial_θ(model, cmdargs)
    if haskey(snakemake.input, "params_restart") &&
        !isempty(snakemake.input["params_restart"])
        fname = snakemake.input["params_restart"]
        params_name = splitext(fname)[1] * cmdargs["prefix"] * ".jld2"
        if isfile(params_name)
            return Float32.(JLD2.load_object(params_name).minimizer)
        end
    end
    base = initial_parameters(model)
    return Float32.(base .+ cmdargs["rand-offset"] .* (rand(length(base)) .- 0.5))
end

# -------------------------------------------------------------------------------------------------
# EstimationState: bundles dense problem objects + per-call buffers + caches.
# Shared by the inner Newton, the Optim outer wrappers, and the KNITRO branch.
# -------------------------------------------------------------------------------------------------

mutable struct EstimationState{M, DM, W, LS, EHW}
    model::M
    dm::DM
    work::W
    θ::Vector{Float32}
    i_δ::Vector{Int}
    i_nonδ::Vector{Int}
    # Per-call buffers (allocated once, reused)
    grad_buf::Vector{Float32}
    bhhh_buf::Matrix{Float32}
    step_buf::Vector{Float32}
    δ_grad_buf::Vector{Float32}
    # Envelope-Hessian workspace and per-call buffer.
    envwork::EHW
    envelope_H_buf::Matrix{Float32}
    # Caches: cached_π / cached_θ_opt are the (π → δ*) map at the latest evaluated point.
    # cached_conv_* hold the convergence-point statistics for the outer optimizer.
    cached_π::Vector{Float32}
    cached_θ_opt::Vector{Float32}
    inner_iter::Base.RefValue{Int}
    last_ll::Base.RefValue{Float32}            # NLL at most recently accepted outer point
    cached_conv_ll::Base.RefValue{Float32}     # NLL at δ*(π) on inner Newton convergence
    cached_conv_grad::Vector{Float32}          # ∇NLL wrt π
    cached_conv_bhhh::Matrix{Float32}
    cached_conv_envelope_H::Matrix{Float32}   # envelope-corrected H̃_NLL (Newton/BFGS use this)
    diag_logged::Base.RefValue{Bool}           # whether the iter-0 BHHH/H̃ diagnostic has been emitted
    # KNITRO --full path: cache the most recently evaluated full θ so the value
    # and gradient callbacks at the same point only trigger one eval_combined! pass.
    last_full_θ::Vector{Float32}
    last_full_ll::Base.RefValue{Float32}
    ls::LS
    nonδ_labels::Vector{String}
end

function EstimationState(model, dm, work, θ::Vector{Float32})
    idx = param_indices(model)
    i_δ = collect(idx.i_δ)
    i_nonδ = collect(idx.i_Πᵦ[1]:idx.i_η[end])
    N_θ = length(θ)
    N_NONδ = length(i_nonδ)
    envwork = EnvelopeHessWork(dm)
    return EstimationState(
        model, dm, work, copy(θ),
        i_δ, i_nonδ,
        zeros(Float32, N_θ),
        zeros(Float32, N_NONδ, N_NONδ),
        zeros(Float32, length(i_δ)),
        zeros(Float32, length(i_δ)),
        envwork,
        zeros(Float32, N_NONδ, N_NONδ),
        fill(NaN32, N_NONδ),
        copy(θ),
        Ref(0),
        Ref(NaN32),
        Ref(NaN32),
        zeros(Float32, N_NONδ),
        1.0f-4 * Matrix{Float32}(I, N_NONδ, N_NONδ),
        1.0f-4 * Matrix{Float32}(I, N_NONδ, N_NONδ),
        Ref(false),
        fill(NaN32, N_θ),
        Ref(NaN32),
        HagerZhang(),
        build_nonδ_labels(),
    )
end

# -------------------------------------------------------------------------------------------------
# Hessian diagnostics — emitted once at the first inner-Newton convergence so we can
# eyeball the magnitude gap between BHHH (∂²NLL/∂π²) and the envelope-corrected
# H̃_NLL = ∂²NLL/∂π² − C̃·(∂²NLL/∂δ²)⁻¹·C̃ᵀ.
# -------------------------------------------------------------------------------------------------

function _eig_extremes(M::AbstractMatrix{Float32})
    Md = Symmetric(Float64.(M))
    return (Float32(eigmin(Md)), Float32(eigmax(Md)))
end

function log_hessian_diagnostics!(
    state::EstimationState, n_singular::Int, shift::Float32, elapsed::Float64
)
    bhhh = state.cached_conv_bhhh
    H̃ = state.cached_conv_envelope_H
    db = diag(bhhh)
    dh = diag(H̃)
    λb_min, λb_max = _eig_extremes(bhhh)
    λh_min, λh_max = _eig_extremes(H̃)
    @info "envelope-Hessian diagnostic (t=$(round(elapsed; sigdigits=3))s, " *
        "singular blocks=$n_singular, λ-shift=$(round(shift; sigdigits=4)))"
    @info "  BHHH  diag: min=$(round(minimum(db); sigdigits=4)) " *
        "max=$(round(maximum(db); sigdigits=4))   λ: $(round(λb_min; sigdigits=4)) … $(round(λb_max; sigdigits=4))"
    @info "  H̃    diag: min=$(round(minimum(dh); sigdigits=4)) " *
        "max=$(round(maximum(dh); sigdigits=4))   λ: $(round(λh_min; sigdigits=4)) … $(round(λh_max; sigdigits=4))"
    @info "  ratio diag(BHHH)/diag(H̃): " *
        "min=$(round(minimum(db ./ max.(dh, 1.0f-12)); sigdigits=4)) " *
        "max=$(round(maximum(db ./ max.(dh, 1.0f-12)); sigdigits=4))"
    return nothing
end

# -------------------------------------------------------------------------------------------------
# Inner Newton over δ (concentrated likelihood).
# On convergence populates state.cached_conv_{ll,grad,bhhh,envelope_H}; leaves cached_conv_ll = NaN on failure.
# -------------------------------------------------------------------------------------------------

function concentrated_θ!(state::EstimationState, π::AbstractVector{Float32})
    if π == state.cached_π
        return state.cached_θ_opt
    end
    state.cached_conv_ll[] = NaN32
    state.inner_iter[] += 1
    id = state.inner_iter[]
    @info "Starting inner optimization #$id"

    θ_tmp = copy(state.θ)
    θ_tmp[state.i_nonδ] .= π
    # Warm start: reuse last δ*
    θ_tmp[state.i_δ] .= state.cached_θ_opt[state.i_δ]

    # Warm start guard
    if !isnan(state.last_ll[]) && state.last_ll[] > 0
        ϕ_warm = -eval_ll!(state.work, state.dm, θ_tmp)
        if ϕ_warm > 1.0f3 * state.last_ll[]
            @info "  inner #$id: warm start NLL=$(round(ϕ_warm; sigdigits=4)) >> prev $(round(state.last_ll[]; sigdigits=4)), resetting δ"
            θ_tmp[state.i_δ] .= state.θ[state.i_δ]
        end
    end

    δ_at_inner_start = copy(θ_tmp[state.i_δ])

    newton_iter = 0
    gnorm = Inf32
    stuck_iters = 0
    unstuck_iters = 0
    trying_to_unstick = false
    bad_descent_count = 0

    while true
        newton_iter += 1
        if newton_iter > 60
            @warn "  inner #$id: exceeded 60 newton iters (gnorm=$gnorm)"
            break
        end

        t_combined = @elapsed begin
            ll = eval_combined!(
                state.work, state.dm, θ_tmp;
                grad = state.grad_buf, bhhh = state.bhhh_buf,
            )
        end
        ϕ0 = -ll                                # NLL
        g_δ_nll = -view(state.grad_buf, state.i_δ)  # ∂NLL/∂δ
        gnorm = Float32(norm(g_δ_nll))

        # Float32 CCP underflow guard — try uniform δ shift, then bail if it fails
        if isapprox(ϕ0, 0)
            @warn "  inner #$id iter $newton_iter: NLL≈0, trying +500 δ shift"
            θ_tmp[state.i_δ] .+= 500.0f0
            ll = eval_ll!(state.work, state.dm, θ_tmp)
            if isapprox(-ll, 0)
                @warn "  inner #$id: δ shift didn't help, breaking"
                break
            end
            continue
        end

        if gnorm ≤ 1.0f-2
            @info "  inner #$id converged in $newton_iter newton iters (gnorm=$gnorm)"
            state.cached_conv_ll[] = ϕ0
            state.cached_conv_grad .= .-state.grad_buf[state.i_nonδ]
            state.cached_conv_bhhh .= state.bhhh_buf
            # Envelope-corrected concentrated Hessian. Reuses forward-pass buffers
            # populated by the just-completed eval_combined!. Falls back to BHHH
            # if the computation throws.
            try
                t_env = @elapsed n_singular = eval_envelope_hessian!(
                    state.work, state.dm, state.envwork;
                    H = state.envelope_H_buf,
                )
                shift = definiteness_shift!(state.envelope_H_buf, 1.0f-3)
                state.cached_conv_envelope_H .= state.envelope_H_buf
                if !state.diag_logged[]
                    log_hessian_diagnostics!(state, n_singular, shift, t_env)
                    state.diag_logged[] = true
                end
            catch e
                @warn "  inner #$id: envelope Hessian failed ($e), falling back to BHHH"
                state.cached_conv_envelope_H .= state.bhhh_buf
            end
            break
        end

        # Woodbury Newton step over δ. woodbury_newton_step! uses ∂ℓ/∂δ (ascent gradient) →
        # ascent direction for ℓ = descent direction for NLL.
        δ_grad_ascent = view(state.grad_buf, state.i_δ)
        woodbury_newton_step!(state.step_buf, state.work, state.dm, δ_grad_ascent)
        d = state.step_buf

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

        δ_cur = copy(θ_tmp[state.i_δ])
        θ_trial = copy(θ_tmp)

        function ϕ(α)
            θ_trial[state.i_δ] .= δ_cur .+ Float32(α) .* d
            return -eval_ll!(state.work, state.dm, θ_trial)
        end
        function dϕ(α)
            θ_trial[state.i_δ] .= δ_cur .+ Float32(α) .* d
            eval_ll_δgrad!(state.work, state.dm, θ_trial; δ_grad = state.δ_grad_buf)
            return -Float32(dot(state.δ_grad_buf, d))
        end
        # Fuse ϕ and dϕ — one forward pass gives both ll and δ-gradient.
        function ϕdϕ(α)
            θ_trial[state.i_δ] .= δ_cur .+ Float32(α) .* d
            ll = eval_ll_δgrad!(state.work, state.dm, θ_trial; δ_grad = state.δ_grad_buf)
            return (-ll, -Float32(dot(state.δ_grad_buf, d)))
        end

        local alpha::Float32, phi_alpha::Float32
        try
            a, p = state.ls(ϕ, dϕ, ϕdϕ, 1.0f0, ϕ0, dϕ0)
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
        θ_tmp[state.i_δ] .+= alpha .* d
    end

    # Unconverged exit penalty: restore δ if gnorm still huge
    if gnorm > 1.0f4
        @warn "  inner #$id: exiting with gnorm=$gnorm > 1e4, restoring start δ"
        θ_tmp[state.i_δ] .= δ_at_inner_start
        # state.cached_conv_ll remains NaN — outer_f detects and returns penalty
    end

    state.cached_π .= π
    state.cached_θ_opt .= θ_tmp
    state.θ .= θ_tmp
    return state.cached_θ_opt
end

# -------------------------------------------------------------------------------------------------
# Optim outer optimizer (nelder-mead / newton / newton-tro / bfgs)
# -------------------------------------------------------------------------------------------------

function select_optim_method(cmdargs, state, π₀, outer_fg!, outer_fgh!, outer_zeroth)
    methodname = cmdargs["%COMMAND%"]
    args = cmdargs[methodname]
    if methodname == "nelder-mead"
        return NelderMead(
            intial_simplex = AffineSimplexer(a = args["a"], b = args["b"])
        ),
        outer_zeroth
    elseif methodname == "newton"
        return Newton(
            alphaguess = InitialStatic(scaled = args["scale"], alpha = args["alpha"])
        ),
        only_fgh!(outer_fgh!)
    elseif methodname == "newton-tro"
        return NewtonTrustRegion(
            initial_delta = args["initial-delta"], eta = 0.01, rho_lower = 0.025
        ),
        only_fgh!(outer_fgh!)
    else  # "bfgs" — seed BFGS's initial inverse Hessian with the envelope-corrected H̃ at π₀
        concentrated_θ!(state, π₀)
        return BFGS(
            initial_invH = _ -> inv(state.cached_conv_envelope_H),
            alphaguess = InitialStatic(scaled = args["scale"], alpha = args["alpha"]),
        ),
        only_fg!(outer_fg!)
    end
end

function outer_optimize(state::EstimationState, π₀, cmdargs)
    _prev_π = copy(π₀)

    function outer_fg!(F, G, π)
        if bounds_violated(π)
            isnothing(G) || (G .= 0)
            return isnothing(F) ? nothing : 1.0f10
        end
        @info format_step_log(state.nonδ_labels, π, _prev_π)
        _prev_π .= π
        concentrated_θ!(state, π)
        ll = state.cached_conv_ll[]
        if isnan(ll) || isapprox(ll, 0)
            isnothing(G) || (G .= 0)
            return isnothing(F) ? nothing : 1.0f10
        end
        state.last_ll[] = ll
        isnothing(G) || (G .= state.cached_conv_grad)
        return isnothing(F) ? nothing : ll
    end

    function outer_fgh!(F, G, H, π)
        if bounds_violated(π)
            isnothing(G) || (G .= 0)
            isnothing(H) || (H .= 0)
            return isnothing(F) ? nothing : 1.0f10
        end
        @info format_step_log(state.nonδ_labels, π, _prev_π)
        _prev_π .= π
        concentrated_θ!(state, π)
        ll = state.cached_conv_ll[]
        if isnan(ll) || isapprox(ll, 0)
            isnothing(G) || (G .= 0)
            isnothing(H) || (H .= 0)
            return isnothing(F) ? nothing : 1.0f10
        end
        state.last_ll[] = ll
        isnothing(G) || (G .= state.cached_conv_grad)
        isnothing(H) || (H .= state.cached_conv_envelope_H)
        return isnothing(F) ? nothing : ll
    end

    function outer_zeroth(π)
        concentrated_θ!(state, π)
        return state.cached_conv_ll[]
    end

    _checkpoint_path =
        splitext(snakemake.output[1])[1] * ".checkpoint" * cmdargs["prefix"] * ".jld2"
    _cb_iter = Ref(0)
    function checkpoint_cb(opt_state)
        _cb_iter[] += 1
        if _cb_iter[] % 25 == 0
            JLD2.save_object(
                _checkpoint_path, (; minimizer = state.θ)
            )
        end
        return false
    end
    options = (
        store_trace = true,
        extended_trace = true,
        show_trace = true,
        g_abstol = 1.0f-3,
        x_abstol = NaN,
        x_reltol = NaN,
        f_abstol = NaN,
        f_reltol = NaN,
        allow_f_increases = true,
        callback = checkpoint_cb,
    )

    method, objective = select_optim_method(
        cmdargs, state, π₀, outer_fg!, outer_fgh!, outer_zeroth
    )
    return optimize(objective, π₀, method, Optim.Options(; options..., iterations = 5000))
end

# -------------------------------------------------------------------------------------------------
# JuMP + KNITRO outer optimizer.
# Wraps concentrated_θ! as a user-defined operator with manually supplied gradient
# (∇NLL from envelope theorem) and Hessian (envelope-corrected H̃_NLL).
# KNITRO is loaded only when this branch is taken — it's not in Project.toml.
# -------------------------------------------------------------------------------------------------

function build_jump_callbacks(state::EstimationState)
    function nll_value(xs::Vararg{T, M}) where {T, M}
        π = Float32.(collect(xs))
        concentrated_θ!(state, π)
        ll = state.cached_conv_ll[]
        if isnan(ll) || isapprox(Float64(ll), 0)
            return T(1.0e10)
        end
        state.last_ll[] = ll
        return T(ll)
    end
    function nll_grad!(g::AbstractVector{T}, xs::Vararg{T, M}) where {T, M}
        π = Float32.(collect(xs))
        concentrated_θ!(state, π)
        ll = state.cached_conv_ll[]
        if isnan(ll) || isapprox(Float64(ll), 0)
            fill!(g, zero(T))
            return nothing
        end
        @inbounds for i in 1:length(g)
            g[i] = T(state.cached_conv_grad[i])
        end
        return nothing
    end
    function nll_hess!(H::AbstractMatrix{T}, xs::Vararg{T, M}) where {T, M}
        π = Float32.(collect(xs))
        concentrated_θ!(state, π)
        ll = state.cached_conv_ll[]
        (isnan(ll) || isapprox(Float64(ll), 0)) && return nothing
        @inbounds for i in 1:size(H, 1), j in 1:i
            H[i, j] = T(state.cached_conv_envelope_H[i, j])
        end
        return nothing
    end
    return nll_value, nll_grad!, nll_hess!
end

# Full-θ callbacks for KNITRO's --full mode. KNITRO drives the joint (δ, π)
# optimization itself; each evaluation is one direct eval_combined! pass — no
# concentrated_θ! / inner Newton. Value and gradient at the same θ share one pass
# via the last_full_θ/last_full_ll cache.
function build_full_jump_callbacks(state::EstimationState)
    N_θ = length(state.θ)
    function eval_full!(θ_in::AbstractVector{Float32})
        if θ_in == state.last_full_θ
            return state.last_full_ll[]
        end
        copyto!(state.θ, θ_in)
        ll = eval_combined!(
            state.work, state.dm, state.θ;
            grad = state.grad_buf, bhhh = state.bhhh_buf,
        )
        state.last_full_ll[] = ll
        copyto!(state.last_full_θ, θ_in)
        return ll
    end
    function nll_value_full(xs::Vararg{T, M}) where {T, M}
        θ_in = Float32.(collect(xs))
        ll = eval_full!(θ_in)
        if isnan(ll) || isapprox(Float64(ll), 0)
            return T(1.0e10)
        end
        return T(-ll)
    end
    function nll_grad_full!(g::AbstractVector{T}, xs::Vararg{T, M}) where {T, M}
        θ_in = Float32.(collect(xs))
        ll = eval_full!(θ_in)
        if isnan(ll) || isapprox(Float64(ll), 0)
            fill!(g, zero(T))
            return nothing
        end
        @inbounds for i in 1:N_θ
            g[i] = T(-state.grad_buf[i])
        end
        return nothing
    end
    return nll_value_full, nll_grad_full!
end

function apply_knitro_options!(m, args)
    JuMP.set_attribute(m, "outlev", args["outlev"])
    JuMP.set_attribute(m, "maxit", args["maxit"])
    JuMP.set_attribute(m, "feastol", args["feastol"])
    JuMP.set_attribute(m, "opttol", args["opttol"])
    JuMP.set_attribute(m, "xtol", args["xtol"])
    JuMP.set_attribute(m, "algorithm", args["algorithm"])
    JuMP.set_attribute(m, "hessopt", args["hessopt"])
    JuMP.set_attribute(m, "honorbnds", args["honorbnds"])
    args["bar-initpt"] >= 0 && JuMP.set_attribute(m, "bar_initpt", args["bar-initpt"])
    return nothing
end

function optimize_knitro(state::EstimationState, π₀, cmdargs)
    args = cmdargs["knitro"]
    if args["full"]
        return optimize_knitro_full(state, cmdargs)
    end
    N = length(state.i_nonδ)
    lb = vcat(fill(-15.0, N - 2), -10.0, -5.0)
    ub = vcat(fill(15.0, N - 2), 10.0, 5.0)

    m = JuMP.Model(KNITRO.Optimizer)
    JuMP.@variable(m, lb[i] <= x[i = 1:N] <= ub[i], start = Float64(π₀[i]))

    nll_value, nll_grad!, nll_hess! = build_jump_callbacks(state)
    if args["no-hessian"]
        JuMP.@operator(m, op_nll, N, nll_value, nll_grad!)
    else
        JuMP.@operator(m, op_nll, N, nll_value, nll_grad!, nll_hess!)
    end
    JuMP.@objective(m, Min, op_nll(x...))

    apply_knitro_options!(m, args)

    @info "Starting KNITRO outer optimization" mode = "concentrated" N_vars = N hessian =
        !args["no-hessian"] algorithm = args["algorithm"] hessopt = args["hessopt"]
    JuMP.optimize!(m)

    π_opt = Float32.(JuMP.value.(x))
    concentrated_θ!(state, π_opt)
    final_ll = state.cached_conv_ll[]
    status = string(JuMP.termination_status(m))
    @info "KNITRO done: status=$status NLL=$(round(Float64(final_ll); sigdigits=6))"

    return (
        minimizer = copy(state.θ),
        value = isnan(final_ll) ? 1.0f10 : final_ll,
        knitro_status = status,
    )
end

# Joint (δ, π) optimization with KNITRO's built-in L-BFGS (hessopt=6).
# eval_combined! produces the full gradient in one pass; the per-iteration cost
# is one forward+backward pass and KNITRO maintains its own curvature memory.
function optimize_knitro_full(state::EstimationState, cmdargs)
    args = cmdargs["knitro"]
    args["hessopt"] = 6   # force L-BFGS; user-provided --hessopt is ignored
    N = length(state.θ)

    lb = fill(-Inf, N)
    ub = fill(Inf, N)
    i_nonδ = state.i_nonδ
    n_nonδ = length(i_nonδ)
    @inbounds for k in 1:(n_nonδ - 2)
        lb[i_nonδ[k]] = -15.0
        ub[i_nonδ[k]] = 15.0
    end
    lb[i_nonδ[end - 1]] = -10.0
    ub[i_nonδ[end - 1]] = 10.0
    lb[i_nonδ[end]] = -5.0
    ub[i_nonδ[end]] = 5.0

    m = JuMP.Model(KNITRO.Optimizer)
    JuMP.@variable(m, lb[i] <= x[i = 1:N] <= ub[i], start = Float64(state.θ[i]))

    nll_value_full, nll_grad_full! = build_full_jump_callbacks(state)
    JuMP.@operator(m, op_nll, N, nll_value_full, nll_grad_full!)
    JuMP.@objective(m, Min, op_nll(x...))

    apply_knitro_options!(m, args)

    @info "Starting KNITRO outer optimization" mode = "full-θ" N_vars = N hessian =
        false algorithm = args["algorithm"] hessopt = args["hessopt"]
    JuMP.optimize!(m)

    copyto!(state.θ, Float32.(JuMP.value.(x)))
    final_ll = -eval_ll!(state.work, state.dm, state.θ)
    status = string(JuMP.termination_status(m))
    @info "KNITRO done: status=$status NLL=$(round(Float64(final_ll); sigdigits=6))"

    return (
        minimizer = copy(state.θ),
        value = isnan(final_ll) ? 1.0f10 : final_ll,
        knitro_status = status,
    )
end

# -------------------------------------------------------------------------------------------------
# Driver
# -------------------------------------------------------------------------------------------------

# Per-block GEMMs have inner dim K+1=5, so BLAS internal threading doesn't help much.
# Julia threads at the block level give the win; set BLAS=1 to avoid oversubscription.
# (The one large GEMM/SYRK in eval_combined! — the 22×22 BHHH and Π_grad — is tiny.)
LinearAlgebra.BLAS.set_num_threads(1)

seed!(78704)
snakemake!(stdin)
const SETTINGS = build_argparse_settings()
model = DemandModel{FixedCoefType}(; JLD2.load_object(snakemake.input["model"])...)

with_logging() do
    cmdargs = parse_args(SETTINGS)
    zips = Registrations.zips().zip_code

    @info "Building DenseModel" Z_zips = length(zips) N = length(param_indices(model).i_δ)
    dm = DenseModel(model, zips)
    work = DenseWork(dm)
    @info "DenseModel built" n_blocks = dm.n_blocks

    θ = build_initial_θ(model, cmdargs)
    state = EstimationState(model, dm, work, θ)
    π₀ = Float32.(θ[state.i_nonδ])

    @info "Starting outer optimization"
    is_knitro = cmdargs["%COMMAND%"] == "knitro"
    result = is_knitro ? optimize_knitro(state, π₀, cmdargs) :
             outer_optimize(state, π₀, cmdargs)

    final_nll = is_knitro ? result.value : result.minimum
    converged_str =
        is_knitro ?
        "status=$(result.knitro_status)" :
        "converged=$(Optim.converged(result))"
    @info "\nEstimation done: NLL=$(round(Float64(final_nll); sigdigits=6)) $converged_str\n"

    JLD2.save_object(snakemake.output[1], result)
end
