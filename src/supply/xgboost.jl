using XGBoost
using Logging: NullLogger, with_logger
using Random
using LogExpFunctions: logistic
using StatsBase: mean

const XGB_DEFAULT_PARAMS = (
    objective = "reg:squarederror",
    base_score = -2.0,
    max_depth = 3,
    eta = 0.15,
    lambda = 1.0,
    min_child_weight = 1.0f0,
    tree_method = "hist",
    subsample = 0.8,
    max_bin = 32,
    seed = 85704,
)

function fit_xgb_custom(
    compute_g_h, dm::XGBoost.DMatrix;
    num_round::Int, params = XGB_DEFAULT_PARAMS,
    xgb_model::Vector{UInt8} = UInt8[],
)
    b = with_logger(NullLogger()) do
        return XGBoost.Booster(dm; params...)
    end
    !isempty(xgb_model) && XGBoost.load!(b, xgb_model)
    for _ in 1:num_round
        ŷ = XGBoost.predict(b, dm; margin = true)
        grad, hess = compute_g_h(ŷ)
        with_logger(NullLogger()) do
            return XGBoost.updateone!(b, dm, grad, hess)
        end
    end
    return b
end

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

function _stratified_kfold(strata::AbstractVector, K::Int, rng::AbstractRNG)
    folds = Vector{Int}(undef, length(strata))
    for s in unique(strata)
        rows = findall(==(s), strata)
        perm = randperm(rng, length(rows))
        for (i, p) in enumerate(perm)
            folds[rows[p]] = ((i - 1) % K) + 1
        end
    end
    return folds
end

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
        return XGBoost.Booster(dm_tr; params...)
    end
    best_val = Inf
    best_round = 0
    stale = 0
    for round in 1:max_round
        ŷ_tr = XGBoost.predict(b, dm_tr; margin = true)
        grad, hess = gh_tr(ŷ_tr)
        with_logger(NullLogger()) do
            return XGBoost.updateone!(b, dm_tr, grad, hess)
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

    dm_full = XGBoost.DMatrix(X)
    XGBoost.setinfo!(dm_full, "label", y)
    gh_full = make_gh(1:n)
    b_final = with_logger(NullLogger()) do
        return XGBoost.Booster(dm_full; params...)
    end
    for _ in 1:best_round
        ŷ = XGBoost.predict(b_final, dm_full; margin = true)
        grad, hess = gh_full(ŷ)
        with_logger(NullLogger()) do
            return XGBoost.updateone!(b_final, dm_full, grad, hess)
        end
    end
    finalize(dm_full)
    return b_final, best_round, best_val
end

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
        return XGBoost.Booster(dm_tr; objective = objective, params...)
    end
    best_val = Inf
    best_round = 0
    stale = 0
    for round in 1:max_round
        with_logger(NullLogger()) do
            return XGBoost.updateone!(b, dm_tr)
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
        return XGBoost.Booster(dm_full; objective = objective, params...)
    end
    for _ in 1:best_round
        with_logger(NullLogger()) do
            return XGBoost.updateone!(b_final, dm_full)
        end
    end
    finalize(dm_full)
    return b_final, best_round
end

const QUANTILE_LEVELS = Float32[
    0.025, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 0.975
]

@inline function pinball_grad_hess(ŷ::Float32, y::Float32, τ::Float32)
    g = ŷ ≥ y ? (1.0f0 - τ) : -τ
    return g, 1.0f0
end

@inline function pinball_nll(ŷ::Float32, y::Float32, τ::Float32)
    d = y - ŷ
    return d ≥ 0 ? τ * d : (τ - 1.0f0) * d
end

function fit_quantile_grid(
    X::AbstractMatrix,
    y::AbstractVector;
    τs::AbstractVector{Float32} = QUANTILE_LEVELS,
    params = XGB_DEFAULT_PARAMS,
    max_round::Int = 500,
    patience::Int = 20,
    strata::Union{Nothing, AbstractVector} = nothing,
    rng::AbstractRNG = Random.default_rng(),
)
    y_f = Float32.(y)
    out = Vector{Vector{UInt8}}(undef, length(τs))
    for (qi, τ) in enumerate(τs)
        base = Float64(_marginal_quantile(y_f, τ))
        τ_c = Float32(τ)
        b, br, _ = fit_xgb_custom_oos(
            idxs ->
                let m = length(idxs), y_s = y_f[idxs]
                    ŷ -> begin
                        grad = Vector{Float32}(undef, m)
                        hess = Vector{Float32}(undef, m)
                        @inbounds for j in 1:m
                            grad[j], hess[j] = pinball_grad_hess(ŷ[j], y_s[j], τ_c)
                        end
                        return grad, hess
                    end
                end,
            idxs ->
                let m = length(idxs), y_s = y_f[idxs]
                    ŷ -> begin
                        nll = 0.0
                        @inbounds for j in 1:m
                            nll += pinball_nll(ŷ[j], y_s[j], τ_c)
                        end
                        return nll / m
                    end
                end,
            X,
            y_f;
            max_round = max_round,
            patience = patience,
            strata = strata,
            rng = rng,
            params = merge(params, (base_score = base,)),
        )
        out[qi] = br == 0 ? UInt8[] : copy(XGBoost.save(b, Vector{UInt8}; format = "ubj"))
        finalize(b)
    end
    return out
end

function _marginal_quantile(y::AbstractVector{Float32}, τ::Float32)
    n = length(y)
    n == 0 && return 0.0f0
    ys = sort(y)
    h = 1 + (n - 1) * τ
    lo = clamp(floor(Int, h), 1, n)
    hi = clamp(lo + 1, 1, n)
    w = Float32(h - lo)
    return ys[lo] * (1 - w) + ys[hi] * w
end

@inline function sample_from_quantile_grid(
    rng::AbstractRNG,
    τs::AbstractVector{Float32} = QUANTILE_LEVELS,
)
    u = Float32(rand(rng))
    return interp_quantile_grid(u, τs)
end

@inline function interp_quantile_grid(
    u::Real,
    τs::AbstractVector{Float32} = QUANTILE_LEVELS,
)
    K = length(τs)
    u ≤ τs[1] && return [(1.0, 1)]
    u ≥ τs[end] && return [(1.0, length(τs))]
    @inbounds for i in 1:(K - 1)
        if u ≤ τs[i + 1]
            w = Float32((u - τs[i]) / (τs[i + 1] - τs[i]))
            return [(1-w, i), (w, i+1)]
        end
    end
end

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

const RHO_MIN = 0.005f0

rho_from_margin(gᵣ::Real) = RHO_MIN + (1 - RHO_MIN) * logistic(gᵣ)

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
