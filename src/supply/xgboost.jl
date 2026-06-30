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
    min_child_weight = 5,
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

"""
    fit_xgb_custom_oos(make_gh, make_nll, X, y; ...) -> (booster, best_round)

Split rows 80/20 by `val_frac`, fit on train while tracking the validation NLL
each round, stop after `patience` non-improving rounds, then *refit* on the full
data for `best_round` iterations. Returns the full-data booster plus the
best-round count.

  - `make_gh(row_idxs)::(ŷ -> (grad::Vector, hess::Vector))` — produces the
    per-subset gradient/Hessian closure with the right per-row context sliced
    to those rows.
  - `make_nll(row_idxs)::(ŷ -> Float64)` — produces the per-subset mean NLL
    closure used as the validation loss.
  - `y` is the (decorative) DMatrix label; closures recover real per-row context.
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
    return b_final, best_round
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

# p_base loss gradient w.r.t. raw output f, given y (sale), z (complier flag),
# and current ρ̃ estimate (held fixed within a sweep).
function pbase_grad_hess(f::Real, y::Real, z::Real, ρ̃::Real)
    p = logistic(f)
    if iszero(z)
        return Float32(p - y), Float32(p * (1 - p))
    end
    u = 1 - ρ̃
    q = p + (1 - p) * ρ̃
    g = (1 - y) * p - y * u * p * (1 - p) / q
    h = u * p^2 * (1 - p) / q
    return Float32(g), Float32(h)
end

function pbase_nll(f::Real, y::Real, z::Real, ρ̃::Real)
    p = clamp(logistic(f), 1.0f-12, 1 - 1.0f-12)
    if iszero(z)
        return -(y * log(p) + (1 - y) * log(1 - p))
    end
    q = clamp(p + (1 - p) * ρ̃, 1.0f-12, 1 - 1.0f-12)
    return -(y * log(q) + (1 - y) * log((1 - p) * (1 - ρ̃) + 1.0f-12))
end

# ρ̃ loss gradient w.r.t. raw output gᵣ. Only applied to complier-classified rows.
function rho_grad_hess(gᵣ::Real, y::Real, p::Real)
    ρ̃ = logistic(gᵣ)
    q = p + (1 - p) * ρ̃
    g = (1 - y) * ρ̃ - y * (1 - p) * ρ̃ * (1 - ρ̃) / q
    h = (1 - p) * ρ̃^2 * (1 - ρ̃) / q
    return Float32(g), Float32(h)
end

function rho_nll(gᵣ::Real, y::Real, p::Real)
    ρ̃ = clamp(logistic(gᵣ), 1.0f-12, 1 - 1.0f-12)
    q = clamp(p + (1 - p) * ρ̃, 1.0f-12, 1 - 1.0f-12)
    return -(y * log(q) + (1 - y) * log((1 - p) * (1 - ρ̃) + 1.0f-12))
end

# Sales-target loss. P(ι = 1 | q, φ) = σ(q − φ). NLL = −log σ(s(q − f))
# where s = 2ι − 1 ∈ {−1, +1}. Gradient s(1 − p), Hessian p(1 − p) with
# p = σ(s(q − f)). Sign on s captures both branches in one closed form.
function phi_grad_hess(f::Real, q::Real, ι::Real)
    s = 2ι - 1
    p = logistic(s * (q - f))
    g = s * (1 - p)
    h = p * (1 - p)
    return Float32(g), Float32(h)
end

function phi_nll(f::Real, q::Real, ι::Real)
    s = 2ι - 1
    return -log(clamp(logistic(s * (q - f)), 1.0f-12, 1 - 1.0f-12))
end
