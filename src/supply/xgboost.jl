#### Custom-objective XGBoost loop and OOS best-round refit protocol.
####
#### Three estimations consume this:
####   compliance_estimation.jl  → fit_xgb_custom (cross-fit step) + fit_xgb_custom_oos (final refit)
####   price_density_estimation  → fit_xgb_oos (stock objective)
####   sales_target_estimation   → fit_xgb_custom_oos (logistic mixture)

using XGBoost
using Logging: NullLogger, with_logger
using Random
using LogExpFunctions: logistic
using StatsBase: mean

const XGB_DEFAULT_PARAMS = (
    # Identity link in XGBoost's view; the custom objectives below re-interpret
    # raw output as logit-space and apply σ in the gradient/Hessian.
    objective = "reg:squarederror",
    # σ(-2) ≈ 0.12 starting prediction; appropriate for both p_base and ρ̃.
    base_score = -2.0,
    max_depth = 3,
    eta = 0.15,
    lambda = 1.0,
    min_child_weight = 1f0,
    tree_method = "hist",
    subsample = 0.8,
    max_bin = 32,
    seed = 85704,
)

#### Custom-objective fitter (fixed round count)

"""
    fit_xgb_custom(compute_g_h, dm; num_round, params = XGB_DEFAULT_PARAMS)

Train an XGBoost Booster for `num_round` iterations using a custom per-row
gradient/Hessian. `compute_g_h(ŷ)::(grad, hess)` evaluates the gradient and
Hessian at the current raw (margin) prediction. Closure-first signature so
callers can use `do`-block syntax.
"""
function fit_xgb_custom(
    compute_g_h, dm::XGBoost.DMatrix; num_round::Int, params = XGB_DEFAULT_PARAMS
)
    b = with_logger(NullLogger()) do
        XGBoost.Booster(dm; params...)
    end
    for _ in 1:num_round
        ŷ = XGBoost.predict(b, dm; margin = true)
        grad, hess = compute_g_h(ŷ)
        with_logger(NullLogger()) do
            XGBoost.updateone!(b, dm, grad, hess)
        end
    end
    return b
end

#### OOS-best-round refit protocol (custom objective)

# Per-stratum 80/20 split. Each stratum contributes at least 1 train + 1 val row
# (strata with <2 rows are placed entirely in train).
function _stratified_split(strata::AbstractVector, val_frac::Float64, rng::AbstractRNG)
    val_idx = Int[]
    tr_idx = Int[]
    for s in unique(strata)
        rows = findall(==(s), strata)
        n_s = length(rows)
        if n_s < 2
            append!(tr_idx, rows)
            continue
        end
        n_v = clamp(floor(Int, val_frac * n_s), 1, n_s - 1)
        perm = randperm(rng, n_s)
        append!(val_idx, rows[perm[1:n_v]])
        append!(tr_idx, rows[perm[(n_v + 1):end]])
    end
    return tr_idx, val_idx
end

"""
    fit_xgb_custom_oos(make_gh, make_nll, X, y; ...) -> (booster, best_round, best_val)

Split rows 80/20 by `val_frac`, fit on train while tracking the validation NLL
each round, stop after `patience` non-improving rounds, then *refit* on the full
data for `best_round` iterations. Returns the full-data booster, the best-round
count, and the best-val NLL (useful for grid-searching link hyperparameters).
When `best_round == 0` the returned booster has no trees and can't be
serialized — callers should check before saving.

  - `make_gh(row_idxs)::(ŷ -> (grad::Vector, hess::Vector))` — produces the
    per-subset gradient/Hessian closure with the right per-row context sliced
    to those rows.
  - `make_nll(row_idxs)::(ŷ -> Float64)` — produces the per-subset mean NLL
    closure used as the validation loss.
  - `y` is the (decorative) DMatrix label; closures recover real per-row context.
  - `strata::Union{Nothing, AbstractVector}` — if provided, val_frac is taken
    from each stratum so the val set is representative of the strata
    distribution (e.g. by ι-quantile bins).
"""
function fit_xgb_custom_oos(
    make_gh,
    make_nll,
    X::AbstractMatrix,
    y::AbstractVector;
    val_frac::Float64 = 0.2,
    max_round::Int = 200,
    patience::Int = 5,
    params = XGB_DEFAULT_PARAMS,
    strata::Union{Nothing, AbstractVector} = nothing,
    rng::AbstractRNG = Random.default_rng(),
)
    n = length(y)
    tr_idx, val_idx = if strata === nothing
        perm = randperm(rng, n)
        n_val = max(1, floor(Int, val_frac * n))
        perm[(n_val + 1):end], perm[1:n_val]
    else
        length(strata) == n ||
            throw(ArgumentError("strata length $(length(strata)) ≠ n $n"))
        _stratified_split(strata, val_frac, rng)
    end

    dm_tr = XGBoost.DMatrix(X[tr_idx, :])
    XGBoost.setinfo!(dm_tr, "label", y[tr_idx])
    dm_val = XGBoost.DMatrix(X[val_idx, :])
    XGBoost.setinfo!(dm_val, "label", y[val_idx])

    gh_tr = make_gh(tr_idx)
    nll_val = make_nll(val_idx)

    b = with_logger(NullLogger()) do
        XGBoost.Booster(dm_tr; params...)
    end
    best_val = Inf
    best_round = 0
    stale = 0
    for round in 1:max_round
        ŷ_tr = XGBoost.predict(b, dm_tr; margin = true)
        grad, hess = gh_tr(ŷ_tr)
        with_logger(NullLogger()) do
            XGBoost.updateone!(b, dm_tr, grad, hess)
        end
        v = nll_val(XGBoost.predict(b, dm_val; margin = true))
        if v < best_val
            best_val = v
            best_round = round
            stale = 0
        else
            stale += 1
            stale >= patience && break
        end
    end
    finalize(b)
    finalize(dm_tr)
    finalize(dm_val)

    # Refit on full data for best_round iterations.
    dm_full = XGBoost.DMatrix(X)
    XGBoost.setinfo!(dm_full, "label", y)
    gh_full = make_gh(1:n)
    b_final = with_logger(NullLogger()) do
        XGBoost.Booster(dm_full; params...)
    end
    for _ in 1:best_round
        ŷ = XGBoost.predict(b_final, dm_full; margin = true)
        grad, hess = gh_full(ŷ)
        with_logger(NullLogger()) do
            XGBoost.updateone!(b_final, dm_full, grad, hess)
        end
    end
    finalize(dm_full)
    return b_final, best_round, best_val
end

#### OOS-best-round refit protocol (stock objective)

"""
    fit_xgb_oos(X, y; objective="reg:squarederror", ...) -> (booster, best_round)

80/20 split + best-round refit for a *stock* XGBoost objective. Default validation
loss is MSE; pass `val_loss(ŷ, y_val)::Float64` to override.
"""
function fit_xgb_oos(
    X::AbstractMatrix,
    y::AbstractVector;
    objective::String = "reg:squarederror",
    val_loss = (ŷ, y_val) -> mean((ŷ .- y_val) .^ 2),
    val_frac::Float64 = 0.2,
    max_round::Int = 1000,
    patience::Int = 5,
    params = NamedTuple(),
    rng::AbstractRNG = Random.default_rng(),
)
    n = length(y)
    perm = randperm(rng, n)
    n_val = max(1, floor(Int, val_frac * n))
    val_idx = perm[1:n_val]
    tr_idx = perm[(n_val + 1):end]

    dm_tr = XGBoost.DMatrix(X[tr_idx, :])
    XGBoost.setinfo!(dm_tr, "label", y[tr_idx])
    dm_val = XGBoost.DMatrix(X[val_idx, :])
    XGBoost.setinfo!(dm_val, "label", y[val_idx])
    y_val_vec = y[val_idx]

    b = with_logger(NullLogger()) do
        XGBoost.Booster(dm_tr; objective = objective, params...)
    end
    best_val = Inf
    best_round = 0
    stale = 0
    for round in 1:max_round
        with_logger(NullLogger()) do
            XGBoost.updateone!(b, dm_tr)
        end
        v = val_loss(XGBoost.predict(b, dm_val), y_val_vec)
        if v < best_val
            best_val = v
            best_round = round
            stale = 0
        else
            stale += 1
            stale >= patience && break
        end
    end
    finalize(b)
    finalize(dm_tr)
    finalize(dm_val)

    dm_full = XGBoost.DMatrix(X)
    XGBoost.setinfo!(dm_full, "label", y)
    b_final = with_logger(NullLogger()) do
        XGBoost.Booster(dm_full; objective = objective, params...)
    end
    for _ in 1:best_round
        with_logger(NullLogger()) do
            XGBoost.updateone!(b_final, dm_full)
        end
    end
    finalize(dm_full)
    return b_final, best_round
end

#### Custom mixture-likelihood gradients / Hessians / NLLs
####
#### All three families: σ-linked raw output, per-row Fisher-information Hessians
#### (always non-negative). Closures pass per-row context (q, ι, z, ρ̃, p) by
#### capture; XGBoost label is decorative.

# p_base loss gradient w.r.t. raw output f, given y (sale), γ (soft compliance
# posterior in [0,1]), and current ρ̃ estimate (held fixed within a sweep).
# Each row's expected-complete-data contribution is γ·complier + (1−γ)·noncomplier.
function pbase_grad_hess(f::Real, y::Real, γ::Real, ρ̃::Real)
    p = logistic(f)
    g_n = p - y
    h_n = p * (1 - p)
    u = 1 - ρ̃
    q = p + (1 - p) * ρ̃
    g_c = (1 - y) * p - y * u * p * (1 - p) / q
    h_c = u * p^2 * (1 - p) / q
    g = γ * g_c + (1 - γ) * g_n
    h = γ * h_c + (1 - γ) * h_n
    return Float32(g), Float32(h)
end

function pbase_nll(f::Real, y::Real, γ::Real, ρ̃::Real)
    p = clamp(logistic(f), 1.0f-12, 1 - 1.0f-12)
    nll_n = -(y * log(p) + (1 - y) * log(1 - p))
    q = clamp(p + (1 - p) * ρ̃, 1.0f-12, 1 - 1.0f-12)
    nll_c = -(y * log(q) + (1 - y) * log((1 - p) * (1 - ρ̃) + 1.0f-12))
    return γ * nll_c + (1 - γ) * nll_n
end

# Floor on ρ̃ via the reparameterization ρ̃ = RHO_MIN + (1−RHO_MIN)·σ(gᵣ).
# The floor separates the complier and noncomplier components structurally:
# a cell with any positive Bernoulli sample z=1 gets at least this much lift
# relative to p_base, so the mixture cannot degenerate into "everyone very
# slightly complier." Consumers convert booster margin → ρ̃ via rho_from_margin.
const RHO_MIN = 0.005f0

"""
    rho_from_margin(gᵣ) -> ρ̃ ∈ [RHO_MIN, 1)

Apply the ρ̃_min-floored logistic link. Used by the M-step's ρ̃ predictions and
the E-step's per-cell ρ̃ lookup so the booster's raw output is interpreted
consistently with `rho_grad_hess` and `rho_nll`.
"""
rho_from_margin(gᵣ::Real) = RHO_MIN + (1 - RHO_MIN) * logistic(gᵣ)

# ρ̃ loss gradient w.r.t. raw output gᵣ under the floored parameterization.
# Chain rule: ∂ρ̃/∂gᵣ = (1 - RHO_MIN)·u·(1-u) where u = σ(gᵣ). The Fisher
# information Hessian carries (∂ρ̃/∂gᵣ)² and stays non-negative.
function rho_grad_hess(gᵣ::Real, y::Real, γ::Real, p::Real)
    u = logistic(gᵣ)
    ρ̃ = RHO_MIN + (1 - RHO_MIN) * u
    q = p + (1 - p) * ρ̃
    dρ̃_dgᵣ = (1 - RHO_MIN) * u * (1 - u)
    g = γ * ((1 - y) / (1 - ρ̃) - y * (1 - p) / q) * dρ̃_dgᵣ
    h = γ * (1 - p) * dρ̃_dgᵣ^2 / (q * (1 - ρ̃))
    return Float32(g), Float32(h)
end

function rho_nll(gᵣ::Real, y::Real, γ::Real, p::Real)
    ρ̃ = clamp(rho_from_margin(gᵣ), 1.0f-12, 1 - 1.0f-12)
    q = clamp(p + (1 - p) * ρ̃, 1.0f-12, 1 - 1.0f-12)
    return γ * -(y * log(q) + (1 - y) * log((1 - p) * (1 - ρ̃) + 1.0f-12))
end

# Sales-target loss with a log link on φ: the booster's raw output is f = log φ,
# so φ = exp(f) is always positive (a sales target can't be negative). ι passed
# as a soft label in [0, 1] (the compliance posterior p_compliant). τ is the
# link temperature on the q scale — caller should pass τ ≈ std(q).
#
#   π = σ((q − exp(f))/τ)
#   NLL = −ι log π − (1 − ι) log(1 − π)
#   dNLL/df = (ι − π) · exp(f) / τ
#   Fisher Hessian = exp(2f) · π(1 − π) / τ²
#
# The Fisher approximation drops the (ι − π) · exp(f)/τ cross-term from the
# exact Hessian — standard practice, since E[ι − π] = 0 at the truth and
# XGBoost wants a non-negative Hessian. The chain-rule exp(f) factor lifts
# the Hessian back to O(1) magnitude, so default λ / min_child_weight work.
function phi_grad_hess(f::Real, q::Real, ι::Real, τ::Real)
    ef = exp(f)
    π = logistic((q - ef) / τ)
    g = (ι - π) * ef / τ
    h = ef * ef * π * (1 - π) / (τ * τ)
    return Float32(g), Float32(h)
end

function phi_nll(f::Real, q::Real, ι::Real, τ::Real)
    π = clamp(logistic((q - exp(f)) / τ), 1.0f-12, 1 - 1.0f-12)
    return -(ι * log(π) + (1 - ι) * log(1 - π))
end
