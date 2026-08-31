using DealerMnA

using LinearAlgebra
using PartialFunctions
using Distributions
import Base.Iterators: product
import Base.Threads: @threads
import ForwardDiff as FD
using StaticArrays
using QuadGK
using LogExpFunctions
using Memoize
using JuMP
import Ipopt

using Plots
using LaTeXStrings

########### PARAMETERS AND SETUP

Dθ = truncated(Exponential(1), lower = 0, upper = 10)
F = cdf$Dθ
f = pdf$Dθ

function D(pL, pH; vL, vH)
    int = quadgk(extrema(Dθ)...) do θ
        uL = θ * vL - pL
        uH = θ * vH - pH
        out = [uL, uH, 0] # soft arg-max
        softmax!(out)
        out .*= f(θ)
        return out[1:2]
    end
    return first(int)
end

function CS(pL, pH; vL, vH)
    ΔH = (pH - pL) / (vH - vL)
    ΔL = pL / vL
    low_surplus = quadgk(ΔL, ΔH) do θ
        (vL - pL / θ) * f(θ)
    end[1]
    high_surplus = quadgk(ΔH, 10) do θ
        (vH - pH / θ) * f(θ)
    end[1]
    return low_surplus + high_surplus
end

@memoize ∂D_cache(vL, vH, T = Float64) = FD.JacobianConfig(splat(D$(; vL, vH)), T[1.0, 2.0])

∂D(pL, pH; vL, vH) = FD.jacobian(splat(D$(; vL, vH)), [pL, pH], ∂D_cache(vL, vH, typeof(pL)))

@memoize ∂D_cache(vL, T = Float64) = FD.GradientConfig(splat(D$(; vL)), T[1.0])

∂D(pL; vL) = FD.gradient(splat(D$(; vL)), [pL], ∂D_cache(vL, typeof(pL)))[1]

function D(pL; vL)
    int = quadgk(extrema(Dθ)...) do θ
        uL = θ * vL - pL
        out = [uL, 0] # soft arg-max
        softmax!(out)
        out .*= f(θ)
        return out[1]
    end
    return first(int)
end

# Max achievable q_L over the feasible price box: pL at its lower bound, pH
# large enough that the H option is never chosen. Used to keep q_bar inside the
# reachable set so the constrained retailer solve stays feasible.
@memoize _qL_max(vL, vH) = D(P_MIN, 100 * vH; vL, vH)[1]

# Max achievable q_L + q_H (both prices at their lower bound). Used as the
# feasible cap for the compliance target φ, which is on total sales.
@memoize _q_total_max(vL, vH) = sum(D(P_MIN, P_MIN; vL, vH))

struct QuadraticCost{T}
    kL::T
    kH::T
end
(c::QuadraticCost)(q) = 0.5 * c.kL * q[1]^2 + 0.5 * c.kH * q[2]^2

struct LinearCost{T <: AbstractVector}
    k::T
end
LinearCost(k...) = LinearCost(collect(k))
LinearCost(k::Real) = LinearCost([k])
(c::LinearCost)(q) = q ⋅ c.k

∂C(c::QuadraticCost, q) = SA[c.kL * q[1], c.kH * q[2]]
∂C(c::LinearCost, q) = SVector{length(c.k)}(c.k)

############### JuMP + IPOPT SETUP

const P_MIN = 1.0e-4
const W_MIN = 1.0e-4
const λ_TOL = 1.0e-6

function make_ipopt_model(; silent = true)
    m = Model(Ipopt.Optimizer)
    silent && set_silent(m)
    set_attribute(m, "hessian_approximation", "limited-memory")
    set_attribute(m, "max_iter", 10_000)
    set_attribute(m, "tol", 1e-6)
    return m
end

function _check_status(m, tag)
    if !is_solved_and_feasible(m, allow_almost = true)
        @error "$tag model solving error:"
        assert_is_solved_and_feasible(m, allow_almost = true)
    end
    return nothing
end

############### RETAILER SOLVERS

# Retailer's profit at (p; C, v). C is LinearCost(w) for a DM retailer, or
# QuadraticCost for the vertically-integrated firm — same solver either way.
_retailer_profit_val(p1, p2, C, vL, vH) = begin
    q = D(p1, p2; vL, vH)
    return p1 * q[1] + p2 * q[2] - C(q)
end

_retailer_profit_grad!(g, p1, p2, C, vL, vH) = begin
    q = D(p1, p2; vL, vH)
    J = ∂D(p1, p2; vL, vH)
    gvec = q + J' * (SA[p1, p2] - ∂C(C, q))
    g[1] = gvec[1]
    g[2] = gvec[2]
    return nothing
end

function solve_retailer(; C, vL, vH, p_init = SA[vL / 2, vH / 2])
    m = make_ipopt_model()
    @variable(m, p[i = 1:2] >= P_MIN, start = p_init[i])
    op = add_nonlinear_operator(
        m, 2,
        (p1, p2) -> _retailer_profit_val(p1, p2, C, vL, vH),
        (g, p1, p2) -> _retailer_profit_grad!(g, p1, p2, C, vL, vH);
        name = :Π_R,
    )
    @objective(m, Max, op(p[1], p[2]))
    optimize!(m)
    _check_status(m, "retailer")
    return SVector{2}(value(p[1]), value(p[2]))
end

# q_L operator (constraint LHS) for the constrained-retailer inner solve.
_qL_val(p1, p2, vL, vH) = D(p1, p2; vL, vH)[1]

_qL_grad!(g, p1, p2, vL, vH) = begin
    J = ∂D(p1, p2; vL, vH)
    g[1] = J[1, 1]
    g[2] = J[1, 2]
    return nothing
end

function solve_constrained_retailer(; C, vL, vH, q_bar, p_init = SA[P_MIN, vH])
    m = make_ipopt_model()
    @variable(m, p[i = 1:2] >= P_MIN, start = p_init[i])
    op_Π = add_nonlinear_operator(
        m, 2,
        (p1, p2) -> _retailer_profit_val(p1, p2, C, vL, vH),
        (g, p1, p2) -> _retailer_profit_grad!(g, p1, p2, C, vL, vH);
        name = :Π_R,
    )
    op_qL = add_nonlinear_operator(
        m, 2,
        (p1, p2) -> _qL_val(p1, p2, vL, vH),
        (g, p1, p2) -> _qL_grad!(g, p1, p2, vL, vH);
        name = :q_L,
    )
    @objective(m, Max, op_Π(p[1], p[2]))
    qcon = @constraint(m, op_qL(p[1], p[2]) >= q_bar)
    optimize!(m)
    _check_status(m, "constrained retailer")
    p_star = SVector{2}(value(p[1]), value(p[2]))
    λ = abs(dual(qcon)) # active-constraint multiplier ≥ 0
    return (p = p_star, λ = λ)
end

# Single-product retailer facing wholesale w_L on the low good only — used to
# evaluate the deviating retailer's profit in the allocation-game IC.
function solve_single_retailer(; wL, vL, p_init = vL / 2)
    m = make_ipopt_model()
    @variable(m, p >= P_MIN, start = p_init)
    op = add_nonlinear_operator(
        m, 1,
        p1 -> (p1 - wL) * D(p1; vL),
        p1 -> D(p1; vL) + (p1 - wL) * ∂D(p1; vL);
        name = :Π_defect,
    )
    @objective(m, Max, op(p))
    optimize!(m)
    _check_status(m, "single-product retailer")
    return value(p)
end

############### RESIDUAL-TRICK RESIDUALS

# Retailer stationarity for linear cost = w: r(p; w) = q(p) + J(p)' * (p − w).
# ForwardDiff on this at the retailer's converged p* gives ∂r/∂p and ∂r/∂w
# for the implicit function theorem: dp*/dw = −(∂r/∂p)⁻¹ · (∂r/∂w).
function _retailer_foc_residual(p1, p2, w1, w2, vL, vH)
    q = D(p1, p2; vL, vH)
    J = ∂D(p1, p2; vL, vH)
    return q + J' * (SA[p1, p2] - SA[w1, w2])
end

# KKT residual for the constrained retailer with the q_L ≥ q_bar constraint
# ACTIVE. Unknowns: (p1, p2, λ). Parameters: (w1, w2, q_bar). Stationarity of
# L = Π_R + λ · (q_L − q_bar) gives q + J'(p − w) + λ · ∇q_L = 0; the third
# equation is the active constraint itself.
function _constr_retailer_kkt(p1, p2, λ, w1, w2, q_bar, vL, vH)
    q = D(p1, p2; vL, vH)
    J = ∂D(p1, p2; vL, vH)
    grad_lag = q + J' * (SA[p1, p2] - SA[w1, w2]) + λ * SA[J[1, 1], J[1, 2]]
    return SA[grad_lag[1], grad_lag[2], q[1] - q_bar]
end

############### MANUFACTURER CALLBACK BUILDERS

# Double-marginalization manufacturer: Π_M(w) = w·q(p*(w)) − C(q(p*(w))).
# Residual trick on the unconstrained retailer FOC.
function _dm_manufacturer_callbacks(C, vL, vH)
    Π_M = function (w1, w2)
        w = SA[w1, w2]
        p = solve_retailer(; C = LinearCost(w), vL, vH)
        q = D(p[1], p[2]; vL, vH)
        return w[1] * q[1] + w[2] * q[2] - C(q)
    end
    ∇Π_M! = function (g, w1, w2)
        w = SA[w1, w2]
        p = solve_retailer(; C = LinearCost(w), vL, vH)
        q = D(p[1], p[2]; vL, vH)
        J = ∂D(p[1], p[2]; vL, vH)
        drdp = FD.jacobian([p[1], p[2]]) do pvec
            return _retailer_foc_residual(pvec[1], pvec[2], w[1], w[2], vL, vH)
        end
        drdw = FD.jacobian([w[1], w[2]]) do wvec
            return _retailer_foc_residual(p[1], p[2], wvec[1], wvec[2], vL, vH)
        end
        dpdw = -drdp \ drdw
        ξ = J' * (w - ∂C(C, q))
        gvec = q + dpdw' * ξ
        g[1] = gvec[1]
        g[2] = gvec[2]
        return nothing
    end
    return Π_M, ∇Π_M!
end

# Allocation-game manufacturer's profit Π_M(w, q_bar). Constrained retailer's
# solution p*(w, q_bar) enters via residual trick on the 3-variable KKT system
# when the sales-target constraint is active; when inactive, dp/dq_bar = 0 and
# dp/dw comes from the unconstrained retailer FOC.
function _alloc_manufacturer_callbacks(C, vL, vH)
    Π_M = function (w1, w2, q_bar)
        w = SA[w1, w2]
        (; p) = solve_constrained_retailer(; C = LinearCost(w), vL, vH, q_bar)
        q = D(p[1], p[2]; vL, vH)
        return w[1] * q[1] + w[2] * q[2] - C(q)
    end
    ∇Π_M! = function (g, w1, w2, q_bar)
        w = SA[w1, w2]
        sol = solve_constrained_retailer(; C = LinearCost(w), vL, vH, q_bar)
        p, λ = sol.p, sol.λ
        q = D(p[1], p[2]; vL, vH)
        J = ∂D(p[1], p[2]; vL, vH)
        ξ = J' * (w - ∂C(C, q))
        if λ > λ_TOL
            drdx = FD.jacobian([p[1], p[2], λ]) do xvec
                return _constr_retailer_kkt(
                    xvec[1], xvec[2], xvec[3], w[1], w[2], q_bar, vL, vH
                )
            end
            drdθ = FD.jacobian([w[1], w[2], q_bar]) do θvec
                return _constr_retailer_kkt(
                    p[1], p[2], λ, θvec[1], θvec[2], θvec[3], vL, vH
                )
            end
            dxdθ = -drdx \ drdθ
            dpdθ = dxdθ[1:2, :]                     # 2×3
            contrib = dpdθ' * ξ                     # 3-vector
            g[1] = q[1] + contrib[1]
            g[2] = q[2] + contrib[2]
            g[3] = contrib[3]                       # ∂Π_M/∂q_bar has no direct term
        else
            drdp = FD.jacobian([p[1], p[2]]) do pvec
                return _retailer_foc_residual(pvec[1], pvec[2], w[1], w[2], vL, vH)
            end
            drdw = FD.jacobian([w[1], w[2]]) do wvec
                return _retailer_foc_residual(p[1], p[2], wvec[1], wvec[2], vL, vH)
            end
            dpdw = -drdp \ drdw                     # 2×2
            contrib = dpdw' * ξ                     # 2-vector
            g[1] = q[1] + contrib[1]
            g[2] = q[2] + contrib[2]
            g[3] = 0.0                              # constraint slack ⇒ q_bar has no local effect
        end
        return nothing
    end
    return Π_M, ∇Π_M!
end

# Retailer's compliance profit Π_R^comply(w, q_bar) = (p − w)·q(p) at the
# constrained retailer's solution. This IS the retailer's own maximization, so
# the envelope theorem gives the gradient directly: ∂L/∂w = −q(p*),
# ∂L/∂q_bar = −λ* (the KKT multiplier on the constraint).
function _alloc_retailer_comply_callbacks(vL, vH)
    Π_R = function (w1, w2, q_bar)
        w = SA[w1, w2]
        (; p) = solve_constrained_retailer(; C = LinearCost(w), vL, vH, q_bar)
        q = D(p[1], p[2]; vL, vH)
        return (p[1] - w[1]) * q[1] + (p[2] - w[2]) * q[2]
    end
    ∇Π_R! = function (g, w1, w2, q_bar)
        w = SA[w1, w2]
        sol = solve_constrained_retailer(; C = LinearCost(w), vL, vH, q_bar)
        q = D(sol.p[1], sol.p[2]; vL, vH)
        g[1] = -q[1]
        g[2] = -q[2]
        g[3] = -sol.λ
        return nothing
    end
    return Π_R, ∇Π_R!
end

# Deviating retailer's profit as a function of w_L only (they drop the high
# good). Envelope again: dΠ_defect/dw_L = −q_L(p_L*).
function _defect_callbacks(vL)
    Π_D = function (wL)
        pL = solve_single_retailer(; wL, vL)
        return (pL - wL) * D(pL; vL)
    end
    ∇Π_D! = function (wL)
        pL = solve_single_retailer(; wL, vL)
        return -D(pL; vL)
    end
    return Π_D, ∇Π_D!
end

############### SUMMARY FUNCTIONS

function vertically_integrated(; C, vL, vH)
    p = solve_retailer(; C, vL, vH)
    q = D(p[1], p[2]; vL, vH)
    Π_manufacturer = p ⋅ q - C(q)  # single firm — no retailer split
    return (; p, q, Π_manufacturer, CS = CS(p[1], p[2]; vL, vH))
end

function simple_manufacturer(; C, vL, vH)
    p_vi = solve_retailer(; C, vL, vH)
    w_init = SA[max(W_MIN, p_vi[1] / 2), max(W_MIN, p_vi[2] / 2)]

    m = make_ipopt_model()
    @variable(m, w[i = 1:2] >= W_MIN, start = w_init[i])
    Π_M_fn, ∇Π_M! = _dm_manufacturer_callbacks(C, vL, vH)
    op = add_nonlinear_operator(m, 2, Π_M_fn, ∇Π_M!; name = :Π_M)
    @objective(m, Max, op(w[1], w[2]))
    optimize!(m)
    _check_status(m, "double-marg manufacturer")

    w = SVector{2}(value(w[1]), value(w[2]))
    p = solve_retailer(; C = LinearCost(w), vL, vH)
    q = D(p[1], p[2]; vL, vH)
    Π_manufacturer = w ⋅ q - C(q)
    Π_retailer = (p - w) ⋅ q
    return (; p, q, w, Π_manufacturer, Π_retailer, CS = CS(p[1], p[2]; vL, vH))
end

# Given a fixed q_bar, solve the manufacturer's problem: pick w to maximize
# Π_M subject to the retailer's IC. Used both by the plotting sweep and,
# equivalently, as the inner sub-problem of the allocation game.
function constrained_manufacturer(; C, vL, vH, q_bar)
    p_vi = solve_retailer(; C, vL, vH)
    w_init = SA[max(W_MIN, p_vi[1] / 2), max(W_MIN, p_vi[2] / 2)]

    m = make_ipopt_model()
    @variable(m, w[i = 1:2] >= W_MIN, start = w_init[i])
    Π_M_fn, ∇Π_M! = _alloc_manufacturer_callbacks(C, vL, vH)
    Π_R_fn, ∇Π_R! = _alloc_retailer_comply_callbacks(vL, vH)
    Π_D_fn, ∇Π_D! = _defect_callbacks(vL)
    op_M = add_nonlinear_operator(m, 3, Π_M_fn, ∇Π_M!; name = :Π_M)
    op_R = add_nonlinear_operator(m, 3, Π_R_fn, ∇Π_R!; name = :Π_R)
    op_D = add_nonlinear_operator(m, 1, Π_D_fn, ∇Π_D!; name = :Π_D)
    @objective(m, Max, op_M(w[1], w[2], q_bar))
    ic = @constraint(m, op_R(w[1], w[2], q_bar) >= op_D(w[1]))
    optimize!(m)
    _check_status(m, "constrained manufacturer q_bar=$q_bar")

    w_star = SVector{2}(value(w[1]), value(w[2]))
    μ = abs(dual(ic))
    return (; w = w_star, μ)
end

function _allocating_manufacturer_at(; C, vL, vH, qL_cap, w_init, q_bar_init)
    m = make_ipopt_model()
    @variable(m, w[i = 1:2] >= W_MIN, start = w_init[i])
    @variable(m, 0 <= q_bar <= qL_cap, start = q_bar_init)
    Π_M_fn, ∇Π_M! = _alloc_manufacturer_callbacks(C, vL, vH)
    Π_R_fn, ∇Π_R! = _alloc_retailer_comply_callbacks(vL, vH)
    Π_D_fn, ∇Π_D! = _defect_callbacks(vL)
    op_M = add_nonlinear_operator(m, 3, Π_M_fn, ∇Π_M!; name = :Π_M)
    op_R = add_nonlinear_operator(m, 3, Π_R_fn, ∇Π_R!; name = :Π_R)
    op_D = add_nonlinear_operator(m, 1, Π_D_fn, ∇Π_D!; name = :Π_D)
    @objective(m, Max, op_M(w[1], w[2], q_bar))
    @constraint(m, op_R(w[1], w[2], q_bar) >= op_D(w[1]))
    optimize!(m)
    _check_status(m, "allocation game")

    w_star = SVector{2}(value(w[1]), value(w[2]))
    q_bar_star = value(q_bar)
    (; p) = solve_constrained_retailer(; C = LinearCost(w_star), vL, vH, q_bar = q_bar_star)
    q = D(p[1], p[2]; vL, vH)
    p_defect = solve_single_retailer(; wL = w_star[1], vL)
    q_defect = D(p_defect; vL)
    Π_defect = (p_defect - w_star[1]) * q_defect
    Π_retailer = (p - w_star) ⋅ q  # retailer complies at equilibrium ⇒ Π_retailer ≥ Π_defect
    Π_manufacturer = w_star ⋅ q - C(q)
    return (;
        w = w_star, p, q, q_bar = q_bar_star,
        Π_manufacturer, Π_retailer, Π_defect, CS = CS(p[1], p[2]; vL, vH),
    )
end

# Multi-start over qualitatively different regions of (w, q_bar) space. The
# outer surface has an IC-constraint ridge that Ipopt can slide into; a single
# start from DM produces isolated local optima that show as "crashes" in
# heatmaps. Dominance check `Π_M^alloc ≥ Π_M^DM − tol` rejects any converged
# point that is worse than DM — setting q_bar = 0 in the alloc game reproduces
# DM exactly, so at the true optimum this must hold, and violation is a
# reliable signal of a local optimum.
const _ALLOC_LOCAL_OPT_TOL = 1.0e-2

function allocating_manufacturer(; C, vL, vH)
    dm = simple_manufacturer(; C, vL, vH)
    vi = vertically_integrated(; C, vL, vH)
    qL_cap = 0.99 * _qL_max(vL, vH)

    starts = [
        (label = "DM",
            w = dm.w,
            q_bar = clamp(max(dm.q[1], 1.0e-3), 0.0, qL_cap)),
        (label = "VI",
            w = SA[max(W_MIN, vi.p[1] / 2), max(W_MIN, vi.p[2] / 2)],
            q_bar = clamp(vi.q[1], 0.0, qL_cap)),
        (label = "midpoint",
            w = SA[max(W_MIN, vL / 4), max(W_MIN, vH / 4)],
            q_bar = qL_cap / 2),
    ]

    best = nothing
    for s in starts
        candidate = try
            _allocating_manufacturer_at(;
                C, vL, vH, qL_cap, w_init = s.w, q_bar_init = s.q_bar
            )
        catch e
            @debug "alloc start failed" start = s.label exception = e
            continue
        end
        if candidate.Π_manufacturer < dm.Π_manufacturer - _ALLOC_LOCAL_OPT_TOL
            @debug "alloc start hit local optimum (Π_M < DM)" start = s.label Π_M =
                candidate.Π_manufacturer Π_M_dm = dm.Π_manufacturer
            continue
        end
        if isnothing(best) || candidate.Π_manufacturer > best.Π_manufacturer
            best = candidate
        end
    end
    isnothing(best) &&
        error("allocating_manufacturer: all starts failed or hit local optima below DM")
    return best
end

############### TWO-PERIOD COMPLIANCE GAME (Design A)
#
# t=1: retailer holds full {L, H} assortment. Chooses (pL, pH). Realized L sales
#      qL^(1) = D_L(pL, pH; vL, vH) + ε with ε ∼ Normal(0, σ); H sales deterministic.
#      Total realized sales are qL^(1) + qH^(1) = D_L + D_H + ε.
# t=2: if qL^(1) + qH^(1) ≥ φ (target is on TOTAL sales), retailer keeps {L, H};
#      else falls back to L only. Prices are the deterministic best-response
#      given assortment access.
# Manufacturer commits (wL, wH, φ) at t=0.
#
# On the equilibrium path with high compliance, the retailer sells both goods in
# both periods — directly comparable to VI and DM. The compliance probability is
# the lever that lets the manufacturer push (pL, pH) down without lowering (wL, wH).

# Pr(qL^(1) + qH^(1) ≥ φ | p) = Pr(ε ≥ φ − D_L(p) − D_H(p)).
Pr_comply(pL, pH, φ; σ, vL, vH) = ccdf(Normal(0, σ), φ - sum(D(pL, pH; vL, vH)))

# t=2 retailer with full assortment, given wholesale w. Returns everything the
# outer manufacturer callback needs: prices, quantities, retailer profit (for
# the retailer's t=1 continuation value), manufacturer profit (for the outer
# objective). CS is wrapped in try/catch: it's only used for post-solve
# reporting, and quadgk over degenerate intervals (which Ipopt reaches when it
# probes extreme wholesale prices) shouldn't crash the outer solve.
function _period2_bonus(w, C, vL, vH)
    p = solve_retailer(; C = LinearCost(w), vL, vH)
    q = D(p[1], p[2]; vL, vH)
    π_R = (SA[p[1], p[2]] - SA[w[1], w[2]]) ⋅ q
    Π_M = SA[w[1], w[2]] ⋅ q - C(q)
    CS_val = try
        CS(p[1], p[2]; vL, vH)
    catch
        NaN
    end
    return (; p, q, π_R, Π_M, CS_val)
end

# t=2 retailer restricted to L only.
function _period2_base(wL, C, vL)
    p = solve_single_retailer(; wL, vL)
    q = D(p; vL)
    π_R = (p - wL) * q
    Π_M = wL * q - C(SA[q, 0.0])
    CS_val = try
        CS_single(p; vL)
    catch
        NaN
    end
    return (; p, q, π_R, Π_M, CS_val)
end

# Consumer surplus when only L is available (threshold-model analogue of the
# two-product CS: same convention as the existing CS(pL, pH; vL, vH)).
function CS_single(pL; vL)
    ΔL = pL / vL
    return quadgk(ΔL, 10) do θ
        (vL - pL / θ) * f(θ)
    end[1]
end

# Retailer's expected value at (pL, pH) given the manufacturer's (w, φ), the noise
# scale σ, discount β, and pre-computed t=2 blocks. π_R^bonus / π_R^base don't
# depend on t=1 prices (only on w), so they enter as passed scalars.
function _compliance_V_R_val(pL, pH, wL, wH, β, φ, σ, vL, vH, π_R_bonus, π_R_base)
    q = D(pL, pH; vL, vH)
    pc = ccdf(Normal(0, σ), φ - (q[1] + q[2]))
    return (pL - wL) * q[1] + (pH - wH) * q[2] +
           β * (pc * π_R_bonus + (1 - pc) * π_R_base)
end

# Retailer's t=1 FOC as a residual vector. Reused by the retailer's IPOPT
# gradient callback and by ForwardDiff in the manufacturer's residual trick.
# ∂Pr_c/∂p_i = pdf(N(0,σ), φ − Σq) · (J[1,i] + J[2,i]) — target on total sales.
function _compliance_retailer_foc(pL, pH, wL, wH, φ, β, σ, vL, vH, ΔπR)
    q = D(pL, pH; vL, vH)
    J = ∂D(pL, pH; vL, vH)
    n_val = pdf(Normal(0, σ), φ - (q[1] + q[2]))
    J_sum = SA[J[1, 1] + J[2, 1], J[1, 2] + J[2, 2]]
    return q + J' * (SA[pL, pH] - SA[wL, wH]) + β * n_val * ΔπR * J_sum
end

function _compliance_V_R_grad!(g, pL, pH, wL, wH, φ, β, σ, vL, vH, ΔπR)
    foc = _compliance_retailer_foc(pL, pH, wL, wH, φ, β, σ, vL, vH, ΔπR)
    g[1] = foc[1]
    g[2] = foc[2]
    return nothing
end

# Solve retailer's t=1 problem: 2-D Ipopt over (pL, pH). Reuses t=2 subsolves
# so callers can access them from the returned struct.
function solve_compliance_retailer(;
    C, vL, vH, wL, wH, φ, β, σ,
    p_init = SA[max(P_MIN, vL / 2), max(P_MIN, vH / 2)],
)
    bonus = _period2_bonus(SA[wL, wH], C, vL, vH)
    base = _period2_base(wL, C, vL)
    ΔπR = bonus.π_R - base.π_R
    m = make_ipopt_model()
    @variable(m, p[i = 1:2] >= P_MIN, start = p_init[i])
    op = add_nonlinear_operator(
        m, 2,
        (pL, pH) -> _compliance_V_R_val(
            pL, pH, wL, wH, β, φ, σ, vL, vH, bonus.π_R, base.π_R
        ),
        (g, pL, pH) -> _compliance_V_R_grad!(g, pL, pH, wL, wH, φ, β, σ, vL, vH, ΔπR);
        name = :V_R,
    )
    @objective(m, Max, op(p[1], p[2]))
    optimize!(m)
    _check_status(m, "compliance retailer")
    p_star = SVector{2}(value(p[1]), value(p[2]))
    q_star = D(p_star[1], p_star[2]; vL, vH)
    pc = ccdf(Normal(0, σ), φ - (q_star[1] + q_star[2]))
    return (; p = p_star, Pr_comply = pc, bonus, base)
end

# Residual for single-product retailer FOC (used by the base-case residual trick).
_base_retailer_foc(p, wL, vL) = D(p; vL) + (p - wL) * ∂D(p; vL)

# dΠ_M^bonus/dw via residual trick on retailer's t=2 FOC. Same pattern as
# _dm_manufacturer_callbacks; extracted so the compliance callback can reuse it.
function _dΠ_M_bonus_dw(w, p, C, vL, vH)
    q = D(p[1], p[2]; vL, vH)
    J = ∂D(p[1], p[2]; vL, vH)
    drdp = FD.jacobian([p[1], p[2]]) do pvec
        return _retailer_foc_residual(pvec[1], pvec[2], w[1], w[2], vL, vH)
    end
    drdw = FD.jacobian([w[1], w[2]]) do wvec
        return _retailer_foc_residual(p[1], p[2], wvec[1], wvec[2], vL, vH)
    end
    dpdw = -drdp \ drdw
    ξ = J' * (SA[w[1], w[2]] - ∂C(C, q))
    return q + dpdw' * ξ
end

# dΠ_M^base/dwL via residual trick on single-product retailer FOC.
function _dΠ_M_base_dwL(wL, p, C, vL)
    q = D(p; vL)
    dqdp = ∂D(p; vL)
    drdp = FD.derivative(pp -> _base_retailer_foc(pp, wL, vL), p)
    drdwL = -dqdp  # ∂/∂wL of (D(p) + (p-wL)·D'(p)) = -D'(p)
    dpdwL = -drdwL / drdp
    kL = C.kL  # QuadraticCost L-slice: marginal cost = kL·q at q_H = 0
    ξ = dqdp * (wL - kL * q)
    return q + dpdwL * ξ
end

# Manufacturer's expected-profit callback and its (wL, wH, φ) gradient. Envelope
# on retailer's t=1 FOC gives dp^*/dθ (2×3); envelope on retailer's t=2 subproblems
# gives dπ_R^{bonus,base}/dw (feeds ∂F/∂w); residual trick on retailer's t=2
# FOCs gives dΠ_M^{bonus,base}/dw. Manufacturer produces to expected demand
# (D_L, D_H) — Jensen's ½·kL·σ² on the L cost is a small constant we drop so
# that (dm.w, φ = 0) exactly reproduces two-period DM and the dominance check
# has a clean anchor.
function _compliance_manufacturer_callbacks(C, vL, vH, β, σ)
    Π_M_val = function (wL, wH, φ)
        sol = solve_compliance_retailer(; C, vL, vH, wL, wH, φ, β, σ)
        q1 = D(sol.p[1], sol.p[2]; vL, vH)
        Π_M_t1 = wL * q1[1] + wH * q1[2] - C(q1)
        return Π_M_t1 +
               β * (sol.Pr_comply * sol.bonus.Π_M + (1 - sol.Pr_comply) * sol.base.Π_M)
    end

    ∇Π_M! = function (g, wL, wH, φ)
        sol = solve_compliance_retailer(; C, vL, vH, wL, wH, φ, β, σ)
        p, Pr_c = sol.p, sol.Pr_comply
        bonus, base = sol.bonus, sol.base
        q1 = D(p[1], p[2]; vL, vH)
        J = ∂D(p[1], p[2]; vL, vH)
        Σq1 = q1[1] + q1[2]
        n_val = pdf(Normal(0, σ), φ - Σq1)
        ΔπR = bonus.π_R - base.π_R
        J_sum = SA[J[1, 1] + J[2, 1], J[1, 2] + J[2, 2]]

        # F_p (2×2) via ForwardDiff on the retailer's FOC at fixed θ (ΔπR held).
        drdp = FD.jacobian([p[1], p[2]]) do pvec
            return _compliance_retailer_foc(
                pvec[1], pvec[2], wL, wH, φ, β, σ, vL, vH, ΔπR
            )
        end

        # F_θ (2×3) analytic. The compliance term in F is β·n_val·ΔπR·J_sum, so
        # its w-derivative goes through ΔπR (envelope: dπ_R^bonus/dwL = -bonus.q[1],
        # dπ_R^base/dwL = -base.q → ∂ΔπR/∂wL = base.q − bonus.q[1]).
        drdwL = SA[-J[1, 1], -J[1, 2]] + β * n_val * (base.q - bonus.q[1]) * J_sum
        drdwH = SA[-J[2, 1], -J[2, 2]] - β * n_val * bonus.q[2] * J_sum
        drdφ = β * (-(φ - Σq1) / σ^2) * n_val * ΔπR * J_sum
        drdθ = hcat(drdwL, drdwH, drdφ)  # 2×3

        dpdθ = -drdp \ drdθ  # 2×3

        # ∂E[Π_M]/∂p (holding θ fixed): t=1 revenue-minus-cost gradient + the
        # compliance-probability channel on t=2.
        df_dp = J' * (SA[wL, wH] - ∂C(C, q1)) +
                β * n_val * (bonus.Π_M - base.Π_M) * J_sum

        # Manufacturer's t=2 profit derivatives via residual trick on t=2 FOCs.
        dΠ_M_bonus = _dΠ_M_bonus_dw(SA[wL, wH], bonus.p, C, vL, vH)
        dΠ_M_base_dwL = _dΠ_M_base_dwL(wL, base.p, C, vL)

        # Direct partials of E[Π_M] holding p fixed. t=1 direct term is q1[k]
        # (∂/∂w_k of w·q1); t=2 goes through the two subgame profits.
        df_dwL_direct = q1[1] + β * (Pr_c * dΠ_M_bonus[1] + (1 - Pr_c) * dΠ_M_base_dwL)
        df_dwH_direct = q1[2] + β * Pr_c * dΠ_M_bonus[2]
        df_dφ_direct = β * (-n_val) * (bonus.Π_M - base.Π_M)

        g[1] = df_dwL_direct + df_dp ⋅ dpdθ[:, 1]
        g[2] = df_dwH_direct + df_dp ⋅ dpdθ[:, 2]
        g[3] = df_dφ_direct + df_dp ⋅ dpdθ[:, 3]
        return nothing
    end

    return Π_M_val, ∇Π_M!
end

function _compliance_manufacturer_at(; C, vL, vH, β, σ, q_cap, w_init, φ_init)
    m = make_ipopt_model()
    w_max = 3 * max(vL, vH)
    @variable(m, W_MIN <= w[i = 1:2] <= w_max, start = clamp(w_init[i], W_MIN, w_max))
    @variable(m, 0 <= φ <= q_cap, start = clamp(φ_init, 0.0, q_cap))
    Π_M_fn, ∇Π_M! = _compliance_manufacturer_callbacks(C, vL, vH, β, σ)
    op = add_nonlinear_operator(m, 3, Π_M_fn, ∇Π_M!; name = :Π_M)
    @objective(m, Max, op(w[1], w[2], φ))
    optimize!(m)
    _check_status(m, "compliance manufacturer")

    w_star = SVector{2}(value(w[1]), value(w[2]))
    φ_star = value(φ)
    sol = solve_compliance_retailer(;
        C, vL, vH, wL = w_star[1], wH = w_star[2], φ = φ_star, β, σ
    )
    p, Pr_c = sol.p, sol.Pr_comply
    bonus, base = sol.bonus, sol.base
    q1 = D(p[1], p[2]; vL, vH)
    Π_M_t1 = w_star[1] * q1[1] + w_star[2] * q1[2] - C(q1)
    Π_manufacturer = Π_M_t1 + β * (Pr_c * bonus.Π_M + (1 - Pr_c) * base.Π_M)
    Π_R_t1 = (p[1] - w_star[1]) * q1[1] + (p[2] - w_star[2]) * q1[2]
    Π_retailer = Π_R_t1 + β * (Pr_c * bonus.π_R + (1 - Pr_c) * base.π_R)
    CS_t1 = CS(p[1], p[2]; vL, vH)
    CS_expected = CS_t1 + β * (Pr_c * bonus.CS_val + (1 - Pr_c) * base.CS_val)
    return (;
        w = w_star, φ = φ_star, p, Pr_comply = Pr_c, q1, bonus, base,
        Π_manufacturer, Π_retailer, CS = CS_expected,
    )
end

# Manufacturer's problem at a fixed φ (optimizes only over (wL, wH)). Used by
# the tracing sweep in main to plot how the optimal wholesale prices respond to
# the target.
function compliance_manufacturer_fixed_phi(;
    C, vL, vH, β, σ, φ, w_init = SA[max(W_MIN, vL / 4), max(W_MIN, vH / 4)]
)
    m = make_ipopt_model()
    @variable(m, w[i = 1:2] >= W_MIN, start = w_init[i])
    Π_M_fn, ∇Π_M! = _compliance_manufacturer_callbacks(C, vL, vH, β, σ)
    g_buf = zeros(3)
    op = add_nonlinear_operator(
        m, 2,
        (wL, wH) -> Π_M_fn(wL, wH, φ),
        (g, wL, wH) -> begin
            ∇Π_M!(g_buf, wL, wH, φ)
            g[1] = g_buf[1]
            g[2] = g_buf[2]
            return nothing
        end;
        name = :Π_M_fixφ,
    )
    @objective(m, Max, op(w[1], w[2]))
    optimize!(m)
    _check_status(m, "compliance manufacturer (fixed φ=$φ)")
    return SVector{2}(value(w[1]), value(w[2]))
end

# Multi-start over qualitatively different (w, φ) regions and pick the best
# converged candidate. With full t=1 assortment, (dm.w, φ = 0) reproduces two-
# period DM exactly, so we anchor with that value and only accept Ipopt
# candidates that beat it — no crash if every start fails or hits a bad
# corner (compliance mechanism inactive at these parameters is a valid result).

# DM anchor: (w = dm.w, φ = 0) — retailer plays DM in both periods (Pr_c ≡ 1
# since Σq_dm > 0 > φ). Constructed by hand rather than via Ipopt so it can't
# fail. Guarantees compliance_manufacturer always returns at least the two-
# period DM outcome, so downstream heatmaps don't crash on inactive-mechanism
# parameters.
function _dm_anchor(dm, β, C, vL, vH)
    bonus = _period2_bonus(dm.w, C, vL, vH)
    base = _period2_base(dm.w[1], C, vL)
    q1 = dm.q
    Π_manufacturer = dm.Π_manufacturer + β * bonus.Π_M
    Π_retailer = dm.Π_retailer + β * bonus.π_R
    CS_expected = CS(dm.p[1], dm.p[2]; vL, vH) + β * bonus.CS_val
    return (;
        w = dm.w, φ = 0.0, p = SVector{2}(dm.p[1], dm.p[2]),
        Pr_comply = 1.0, q1, bonus, base,
        Π_manufacturer, Π_retailer, CS = CS_expected,
    )
end

function compliance_manufacturer(; C, vL, vH, β, σ)
    dm = simple_manufacturer(; C, vL, vH)
    vi = vertically_integrated(; C, vL, vH)
    q_cap = 0.99 * _q_total_max(vL, vH)
    two_period_dm = (1 + β) * dm.Π_manufacturer

    # Anchor: (dm.w, φ = 0). Always feasible, always returns exactly (1+β)·dm.Π_M.
    dm_anchor = _dm_anchor(dm, β, C, vL, vH)
    @info "compliance dm_anchor" Π_M = dm_anchor.Π_manufacturer two_period_dm

    # Compliance-active start: φ set so retailer's DM equilibrium sits at
    # Pr_c ≈ 0.5, giving n_val a meaningful magnitude for the outer gradient.
    Σq_dm = dm.q[1] + dm.q[2]
    φ_active = clamp(Σq_dm, 1.0e-4, q_cap)
    starts = [
        (label = "DM_no_target", w = dm.w, φ = 1.0e-4),
        (label = "comply_active", w = dm.w, φ = φ_active),
        (label = "VI_halved",
            w = SA[max(W_MIN, vi.p[1] / 2), max(W_MIN, vi.p[2] / 2)],
            φ = 0.3 * q_cap),
        (label = "aggressive_wL",
            w = SA[max(W_MIN, dm.w[1] / 4), dm.w[2]],
            φ = 0.5 * q_cap),
        (label = "midpoint",
            w = SA[max(W_MIN, vL / 4), max(W_MIN, vH / 4)],
            φ = 0.5 * q_cap),
    ]

    best = dm_anchor
    best_label = "dm_anchor"
    for s in starts
        candidate = try
            _compliance_manufacturer_at(;
                C, vL, vH, β, σ, q_cap, w_init = s.w, φ_init = s.φ
            )
        catch e
            @info "compliance start failed" start = s.label exception = e
            continue
        end
        @info "compliance start converged" start = s.label Π_M = candidate.Π_manufacturer φ =
            candidate.φ wL = candidate.w[1] wH = candidate.w[2] Pr_c = candidate.Pr_comply
        if candidate.Π_manufacturer > best.Π_manufacturer
            best = candidate
            best_label = s.label
        end
    end
    @info "compliance best" start = best_label Π_M = best.Π_manufacturer bound = two_period_dm
    return best
end

recovered_surplus(d::Dict) = recovered_surplus(d["c_L"], d["c_H"], d["v_L"], d["v_H"])

function recovered_surplus(cL, cH, vL, vH)
    C = QuadraticCost(cL, cH)
    vi = vertically_integrated(; C, vL, vH)
    dm = simple_manufacturer(; C, vL, vH)
    alloc = allocating_manufacturer(; C, vL, vH)
    vi_ps = vi.Π_manufacturer
    dm_ps = dm.Π_manufacturer + dm.Π_retailer
    alloc_ps = alloc.Π_manufacturer + alloc.Π_retailer
    return (; vi, dm, alloc, recovered = (alloc_ps - dm_ps) / (vi_ps - dm_ps))
end

# Two-period analogue for the compliance game. VI and DM baselines are scaled
# by (1 + β) because those regimes' single-period surplus repeats twice; the
# compliance regime's Π_M / Π_R already include the β-discounted t=2 terms.
function recovered_surplus_compliance(d::Dict)
    return recovered_surplus_compliance(
        d["c_L"], d["c_H"], d["v_L"], d["v_H"], d["β"], d["σ"]
    )
end

function recovered_surplus_compliance(cL, cH, vL, vH, β, σ)
    C = QuadraticCost(cL, cH)
    vi = vertically_integrated(; C, vL, vH)
    dm = simple_manufacturer(; C, vL, vH)
    compliance = compliance_manufacturer(; C, vL, vH, β, σ)
    vi_ps = (1 + β) * vi.Π_manufacturer
    dm_ps = (1 + β) * (dm.Π_manufacturer + dm.Π_retailer)
    comp_ps = compliance.Π_manufacturer + compliance.Π_retailer
    return (; vi, dm, compliance, recovered = (comp_ps - dm_ps) / (vi_ps - dm_ps))
end

############### COMPARATIVE STATICS

# Producer surplus in each regime (VI is single-firm; DM/alloc split into M+R).
_ps(r) = hasproperty(r, :Π_retailer) ? r.Π_manufacturer + r.Π_retailer : r.Π_manufacturer

# Envelope-theorem derivatives of each regime's PS w.r.t. each parameter, at
# equilibrium. Costs are analytic: ∂PS/∂c_k = −½ q_k² because c only enters the
# objective through C(q), so envelope on the outer optimizer's choices leaves
# only the direct cost-function derivative — and this holds identically across
# VI, DM, and alloc regardless of which agent is choosing what. Value
# parameters (v_L, v_H) enter through demand, which shifts equilibrium
# quantities; Ipopt in the DM/alloc solvers blocks AD, so we central-difference
# the outer solver. FD noise here is bounded by Ipopt's tolerance rather than
# by the roughness of the outer optimization landscape, so it's much cleaner
# than raw FD on the recovered-surplus function.
function envelope_derivatives(; cL, cH, vL, vH, h = 1.0e-3)
    C = QuadraticCost(cL, cH)
    vi = vertically_integrated(; C, vL, vH)
    dm = simple_manufacturer(; C, vL, vH)
    alloc = allocating_manufacturer(; C, vL, vH)

    d_cL = (vi = -0.5 * vi.q[1]^2, dm = -0.5 * dm.q[1]^2, alloc = -0.5 * alloc.q[1]^2)
    d_cH = (vi = -0.5 * vi.q[2]^2, dm = -0.5 * dm.q[2]^2, alloc = -0.5 * alloc.q[2]^2)

    fd(f, x) = (f(x + h) - f(x - h)) / (2h)
    d_vL_of(regime) = fd(x -> _ps(regime(; C, vL = x, vH)), vL)
    d_vH_of(regime) = fd(x -> _ps(regime(; C, vL, vH = x)), vH)

    d_vL = (
        vi = d_vL_of(vertically_integrated),
        dm = d_vL_of(simple_manufacturer),
        alloc = d_vL_of(allocating_manufacturer),
    )
    d_vH = (
        vi = d_vH_of(vertically_integrated),
        dm = d_vH_of(simple_manufacturer),
        alloc = d_vH_of(allocating_manufacturer),
    )

    # Quotient rule on recovered = (alloc − dm) / (vi − dm).
    num = _ps(alloc) - _ps(dm)
    denom = _ps(vi) - _ps(dm)
    dR(d) = ((d.alloc - d.dm) * denom - num * (d.vi - d.dm)) / denom^2
    d_recovered = (cL = dR(d_cL), cH = dR(d_cH), vL = dR(d_vL), vH = dR(d_vH))

    return (; d_cL, d_cH, d_vL, d_vH, d_recovered)
end

############### 2D HEATMAPS OF RECOVERED SURPLUS

# Recovered-surplus fraction over a 2D grid of two parameters, holding the
# other two at their baseline. Missing values mark grid points where the outer
# solver failed (e.g. IC infeasibility or Ipopt divergence at extreme params).
function recovered_surplus_grid(; xkey, xrange, ykey, yrange, base, fn = recovered_surplus)
    grid = Matrix{Union{Missing, Float64}}(missing, length(yrange), length(xrange))
    n_fail = 0
    for j in eachindex(xrange), i in eachindex(yrange)
        p = copy(base)
        p[xkey] = xrange[j]
        p[ykey] = yrange[i]
        try
            grid[i, j] = fn(p).recovered
        catch e
            n_fail += 1
            @info "grid point failed" xkey xval = xrange[j] ykey yval = yrange[i] exception = e
        end
    end
    n_fail > 0 && @warn "recovered_surplus_grid: $n_fail / $(length(grid)) points failed" xkey ykey
    return grid
end

function heatmap_recovered(;
    xkey, xrange, ykey, yrange, base, xlab, ylab,
    fn = recovered_surplus, title = "Recovered surplus fraction",
)
    grid = recovered_surplus_grid(; xkey, xrange, ykey, yrange, base, fn)
    plt = heatmap(
        xrange, yrange, grid;
        xlabel = xlab, ylabel = ylab,
        title = title,
        color = :viridis,
    )
    scatter!(plt, [base[xkey]], [base[ykey]];
        label = "baseline", markershape = :star5, markercolor = :red, markersize = 8)
    return plt
end

################ MAIN
function (@main)(_)
    snakemake!()

    RESOLUTION = 50

    with_logging() do
        data = Dict(
            "c_L" => 2.0,
            "c_H" => 2.5,
            "v_L" => 2.5,
            "v_H" => 4.0,
            "β" => 1.0,
            "σ" => 0.05,
        )

        result = recovered_surplus(data)
        derivatives = envelope_derivatives(;
            cL = data["c_L"], cH = data["c_H"], vL = data["v_L"], vH = data["v_H"]
        )
        @info "Envelope derivatives" derivatives

        @info "Compliance regime"
        result_compliance = recovered_surplus_compliance(data)
        @info "Compliance summary" wL_compliance = result_compliance.compliance.w[1] wL_dm = result_compliance.dm.w[1] wH_compliance = result_compliance.compliance.w[2] wH_dm = result_compliance.dm.w[2] φ =
            result_compliance.compliance.φ Pr_c =
            result_compliance.compliance.Pr_comply recovered = result_compliance.recovered
        JLD2.save_object(snakemake.output["jld2"], (; result, derivatives, result_compliance))

        # 2D heatmaps of recovered surplus over pairs of parameters.
        #         HEAT_N = 15
        #         makerange(k, w) = range(max(0.05, data[k] - w), data[k] + w, length = HEAT_N)
        #         pairs = [
        #             (:c_L_c_H, "c_L", "c_H", makerange("c_L", 1.5), makerange("c_H", 1.5),
        #                 L"c_L", L"c_H"),
        #             (:v_L_v_H, "v_L", "v_H", makerange("v_L", 1.5), makerange("v_H", 2.0),
        #                 L"v_L", L"v_H"),
        #             (:c_L_v_L, "c_L", "v_L", makerange("c_L", 1.5), makerange("v_L", 1.5),
        #                 L"c_L", L"v_L"),
        #             (:c_H_v_H, "c_H", "v_H", makerange("c_H", 1.5), makerange("v_H", 2.0),
        #                 L"c_H", L"v_H"),
        #         ]
        #         for (key, xkey, ykey, xr, yr, xlab, ylab) in pairs
        #             @info "Heatmap" xkey ykey
        #             plt = heatmap_recovered(;
        #                 xkey, xrange = xr, ykey, yrange = yr, base = data, xlab, ylab
        #             )
        #             png(plt, snakemake.output["png_$key"])
        #         end
        # 
        #         for (key, xkey, ykey, xr, yr, xlab, ylab) in pairs
        #             @info "Compliance heatmap" xkey ykey
        #             plt = heatmap_recovered(;
        #                 xkey, xrange = xr, ykey, yrange = yr, base = data, xlab, ylab,
        #                 fn = recovered_surplus_compliance,
        #                 title = "Recovered surplus fraction (compliance regime)",
        #             )
        #             png(plt, snakemake.output["png_compliance_$key"])
        #         end

        @info "Tracing profit plot"
        (; vi, dm, alloc) = recovered_surplus(data)
        q_grid = range(0, 0.5, length = 100)
        C = QuadraticCost(data["c_L"], data["c_H"])
        vL, vH = data["v_L"], data["v_H"]
        y = map(q_grid) do q_bar
            try
                cm = constrained_manufacturer(; C, vL, vH, q_bar)
                w = cm.w
                μ = cm.μ
                cr = solve_constrained_retailer(; C = LinearCost(w), vL, vH, q_bar)
                p = cr.p
                λ = cr.λ
                q = D(p[1], p[2]; vL, vH)
                return (;
                    Π_manufacturer = w ⋅ q - C(q),
                    Π_retailer = (p - w) ⋅ q,
                    w,
                    p,
                    q,
                    λ,
                    μ,
                    q_bar,
                    CS = CS(p[1], p[2]; vL, vH),
                )
            catch
                return missing
            end
        end

        y = DataFrame(filter(!ismissing, y))

        plot(
            y.q_bar,
            y.Π_manufacturer,
            xlabel = L"\underbar{q}_L",
            ylabel = L"\pi^M",
            label = "Profit with Sales Target",
            legend = :topright,
        )
        vline!([alloc.q_bar], label = "Optimal Sales Target")
        vline!([vi.q[1]], label = "Vertically Integrated Output")
        vline!([dm.q[1]], label = "Double Marginalization Output")
        png(snakemake.output["profit_plot"])

        plot(
            y.q_bar,
            y.μ,
            xlabel = L"\underbar{q}_L",
            ylabel = L"\mu",
            label = "Lagrange Multiplier on Retailer Incentive Constraint",
            legend = :topright,
        )
        vline!([alloc.q_bar], label = "Optimal Sales Target")
        vline!([dm.q[1]], label = "Double Marginalization Output")
        png(snakemake.output["mu_plot"])

        plot(
            y.q_bar,
            y.λ,
            xlabel = L"\underbar{q}_L",
            ylabel = L"\lambda",
            label = "Lagrange Multiplier on Sales Target Constraint",
            legend = :right,
        )
        vline!([alloc.q_bar], label = "Optimal Sales Target")
        vline!([dm.q[1]], label = "Double Marginalization Output")
        png(snakemake.output["lambda_plot"])

        plot(
            y.q_bar,
            first.(y.w),
            xlabel = L"\underbar{q}_L",
            ylabel = L"w_L",
            label = "Wholesale Price of Low-Quality Product",
            legend = :topright,
        )
        vline!([alloc.q_bar], label = "Optimal Sales Target")
        png(snakemake.output["wL_plot"])

        plot(
            y.q_bar,
            last.(y.w),
            xlabel = L"\underbar{q}_L",
            ylabel = L"w_H",
            label = "Wholesale Price of High-Quality Product",
            legend = :bottomright,
        )
        vline!([alloc.q_bar], label = "Optimal Sales Target")
        png(snakemake.output["wH_plot"])

        plot(
            y.q_bar,
            y.Π_retailer,
            xlabel = L"\underbar{q}_L",
            ylabel = L"\pi^R",
            label = "Profit of Low-Quality Product",
            legend = :bottomright,
        )
        vline!([alloc.q_bar], label = "Optimal Sales Target")
        png(snakemake.output["piR_plot"])

        plot(
            y.q_bar,
            y.Π_manufacturer .+ y.Π_retailer,
            xlabel = L"\underbar{q}_L",
            ylabel = "Total Surplus",
            label = "Total Surplus",
            legend = :bottomright,
        )
        vline!([alloc.q_bar], label = "Optimal Sales Target")
        png(snakemake.output["surplus_plot"])

        plot(
            y.q_bar,
            last.(y.q),
            xlabel = L"\underbar{q}_L",
            ylabel = L"q_H",
            label = "High-Quality Product Quantity",
            legend = :bottomright,
        )
        vline!([alloc.q_bar], label = "Optimal Sales Target")
        png(snakemake.output["qH_plot"])

        # ---- Compliance-regime tracing: sweep φ, re-solve manufacturer's
        # optimal (wL, wH) at each φ, and plot how prices / compliance move.
        @info "Compliance tracing plots"
        β, σ = data["β"], data["σ"]
        q_cap = 0.99 * _q_total_max(vL, vH)
        φ_grid = range(0, q_cap, length = 60)
        comp = result_compliance.compliance
        dm_ref = result_compliance.dm

        yc = map(φ_grid) do φ
            try
                w = compliance_manufacturer_fixed_phi(; C, vL, vH, β, σ, φ)
                sol = solve_compliance_retailer(;
                    C, vL, vH, wL = w[1], wH = w[2], φ, β, σ
                )
                q1 = D(sol.p[1], sol.p[2]; vL, vH)
                Π_M_t1 = w[1] * q1[1] + w[2] * q1[2] - C(q1)
                Π_M_expected =
                    Π_M_t1 +
                    β * (sol.Pr_comply * sol.bonus.Π_M +
                         (1 - sol.Pr_comply) * sol.base.Π_M)
                return (;
                    φ, wL = w[1], wH = w[2], pL = sol.p[1], pH = sol.p[2],
                    qL = q1[1], qH = q1[2],
                    Pr_c = sol.Pr_comply, Π_M = Π_M_expected,
                )
            catch e
                @info "compliance tracing point failed" φ exception = e
                return missing
            end
        end
        yc = DataFrame(filter(!ismissing, yc))

        plot(
            yc.φ, yc.wL,
            xlabel = L"\varphi",
            ylabel = L"w_L^*(\varphi)",
            label = "compliance",
            legend = :topright,
        )
        hline!([dm_ref.w[1]], label = "DM baseline", linestyle = :dash)
        vline!([comp.φ], label = "Optimal target")
        png(snakemake.output["compliance_wL_plot"])

        plot(
            yc.φ, yc.wH,
            xlabel = L"\varphi",
            ylabel = L"w_H^*(\varphi)",
            label = "compliance",
            legend = :bottomright,
        )
        hline!([dm_ref.w[2]], label = "DM baseline", linestyle = :dash)
        vline!([comp.φ], label = "Optimal target")
        png(snakemake.output["compliance_wH_plot"])

        plot(
            yc.φ, yc.Pr_c,
            xlabel = L"\varphi",
            ylabel = L"\Pr(\mathrm{comply})",
            label = "compliance",
            legend = :topright,
        )
        vline!([comp.φ], label = "Optimal target")
        png(snakemake.output["compliance_prcomply_plot"])

        plot(
            yc.φ, yc.Π_M,
            xlabel = L"\varphi",
            ylabel = L"E[\Pi^M](\varphi)",
            label = "compliance",
            legend = :bottomright,
        )
        hline!([(1 + β) * dm_ref.Π_manufacturer], label = "(1+β)·DM baseline", linestyle = :dash)
        vline!([comp.φ], label = "Optimal target")
        png(snakemake.output["compliance_phi_plot"])
    end
    return nothing
end
