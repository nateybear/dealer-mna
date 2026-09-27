#### Residual demand summaries

using Distributions
using DataFrames
using DataStructures

const _PRICE_SEEDS = Ref{Matrix{Float32}}()
const SIM_SIZE = 50

# Coordinated uniform price multipliers applied to a dealer's whole assortment
# in BBL forward simulation. Single source of truth — bbl_forward_simulation
# emits `length(COORDINATED_SHIFTS)` scenarios, and bbl_matrix / bbl_estimation
# derive N_SHIFTS from `length(Supply.COORDINATED_SHIFTS)` so the counts can't
# drift.
const COORDINATED_SHIFTS = Float32[0.75, 0.9, 0.95, 0.98, 0.99, 1.01, 1.02, 1.05, 1.1, 1.25]

"""
    tf_to_j_lookup(model_or_cf) → OrderedDict{(QuarterKey, UUID), Vector{VehicleKey}}

Live products per (quarter, dealer_uid), sorted within each (t, f) by
`k_local` so the order matches `expected_demand` / `dealer_block_locals`.
Only entries whose δ_idx is still in `model.tjf_to_block_k` are returned —
ModelDelete drops products from the view, ModelAdd would grow it.

Memoized on the `DemandModel`. Since ModelAdd/ModelDelete mutate the model
in place, callers who mutate must invalidate the cache via
`invalidate_tf_to_j_lookup!()` before the next query. `residual_demand`
does this at entry and after undoing internal deletes.
"""
@memoize function tf_to_j_lookup(model::DemandModel)
    (; tjf) = model.product_lookup
    grouped = OrderedDict{Tuple{QuarterKey, UUID}, Vector{Tuple{Int, VehicleKey}}}()
    for ((t, j, f), δ_idx) in tjf
        loc = get(model.tjf_to_block_k, δ_idx, nothing)
        loc === nothing && continue
        _, k_local = loc
        key = (t, f)
        if haskey(grouped, key)
            push!(grouped[key], (Int(k_local), j))
        else
            grouped[key] = [(Int(k_local), j)]
        end
    end
    out = OrderedDict{Tuple{QuarterKey, UUID}, Vector{VehicleKey}}()
    for (key, pairs_) in grouped
        sort!(pairs_; by = first)
        out[key] = [p[2] for p in pairs_]
    end
    return out
end

tf_to_j_lookup(cf::Counterfactual) = tf_to_j_lookup(cf.model)

"""
    invalidate_tf_to_j_lookup!()

Empty the `tf_to_j_lookup` memoize cache so the next call rebuilds from the
current model state. Call this after any ModelAdd / ModelDelete / undo that
changed which products live in `model.tjf_to_block_k`.
"""
invalidate_tf_to_j_lookup!() = empty!(Memoize.memoize_cache(tf_to_j_lookup))

function seed!(cf::Counterfactual)
    num_models = length(keys(cf.model.product_lookup.j))
    if !isdefined(_PRICE_SEEDS, 1) || Base.size(_PRICE_SEEDS[], 1) != num_models
        D = Uniform(0.85, 1.15)
        _PRICE_SEEDS[] = rand(D, num_models, SIM_SIZE)
    end
end

"""
    residual_demand(cf, dealer_uid, quarter; return_draws=false, keep=nothing, products=nothing)

Perturbs SIM_SIZE prices for the dealer's assortment at `(quarter, dealer_uid)`
and OLS-fits per-product intercept + own-/cross-price coefficients. Order is
`tf_to_j_lookup`'s (k_local-sorted), so the internal PriceMultiply loop and
`expected_demand` return aligned vectors.

Reads the assortment via `tf_to_j_lookup(cf)` after invalidating the memoize
cache, so tweaks applied by the caller before the call are reflected. Pass
`keep::Set{VehicleKey}` to restrict the perturbed assortment to those VKs;
non-kept products are ModelDelete'd here and restored before returning. Pass
`products::AbstractVector{VehicleKey}` to fix the row/column ordering of the
returned matrix — the caller can then interpret slice indices in a stable
kept-VK layout independent of `keys(product_lookup.j)`'s Dict order. Defaults
to `collect(keys(product_lookup.j))` for backward compat.
"""
function residual_demand(
    cf::Counterfactual, dealer_uid::UUID, quarter;
    return_draws::Bool = false,
    keep::Union{Nothing, Set{VehicleKey}} = nothing,
    products::Union{Nothing, AbstractVector{VehicleKey}} = nothing,
)
    invalidate_tf_to_j_lookup!()
    products = products === nothing ? collect(keys(cf.model.product_lookup.j)) : products
    prod_pos = Dict(p => i for (i, p) in enumerate(products))
    out = fill(NaN32, (length(products), length(products) + 1))

    delete_handles = NamedTuple[]
    if keep !== nothing
        for j in get(tf_to_j_lookup(cf), (quarter, dealer_uid), VehicleKey[])
            j ∈ keep && continue
            h = apply!(cf, ModelDelete(vehicle = j, quarter = quarter, dealer_uid = dealer_uid))
            ismissing(h) || push!(delete_handles, h)
        end
        isempty(delete_handles) || invalidate_tf_to_j_lookup!()
    end
    # If the (t, f) has no live products (empty original assortment, or all
    # non-kept and just ModelDelete'd), the SIM_SIZE loop degenerates and `out`
    # stays all-NaN. Callers should treat NaN as "dealer sat out."
    carried_products = get(tf_to_j_lookup(cf), (quarter, dealer_uid), VehicleKey[])

    prices = zeros(SIM_SIZE, length(carried_products))
    demands = zeros(SIM_SIZE, length(carried_products))
    product_idxs = [prod_pos[j] for j in carried_products]
    seed!(cf)
    for i in 1:SIM_SIZE
        handles = map(product_idxs) do j
            tweak = PriceMultiply(;
                dealer_uid, quarter, vehicle = products[j], by = _PRICE_SEEDS[][j, i]
            )
            apply!(cf, tweak)
        end
        prices[i, :] .= [only(h.old_prices) * h.by for h in handles]
        demands[i, :] .= expected_demand(cf, dealer_uid, quarter)
        for h in handles
            undo!(cf, h)
        end
    end

    # Snapshot the raw price/demand levels before the log transform, in case
    # the caller wants them for BBL moment formation.
    raw_prices = return_draws ? copy(prices) : nothing
    raw_demands = return_draws ? copy(demands) : nothing

    demands .= log.(demands)
    prices .= log.(prices)

    # Preallocate the design matrix outside the per-product loop. Column 1 is the
    # intercept (constant 1s); columns 2:(N_d+1) hold either log(p_k) (k = focal)
    # or log(p_k)·log(p_focal) (k ≠ focal). Only the interacted columns change
    # between iterations, so we scribble in-place.
    N_d = length(product_idxs)
    X = Matrix{Float64}(undef, SIM_SIZE, N_d + 1)
    fill!(view(X, :, 1), 1.0)
    for (i, j) in enumerate(product_idxs)
        lp_i = view(prices, :, i)
        @inbounds for k in 1:N_d
            col = view(X, :, k + 1)
            if k == i
                @simd for r in 1:SIM_SIZE
                    col[r] = prices[r, k]
                end
            else
                @simd for r in 1:SIM_SIZE
                    col[r] = prices[r, k] * lp_i[r]
                end
            end
        end
        α = X \ view(demands, :, i)
        out[j, 1] = α[1]
        @inbounds for (kk, jj) in enumerate(product_idxs)
            out[j, jj + 1] = α[kk + 1]
        end
    end

    # Restore internal ModelDeletes (LIFO) and re-invalidate so anyone
    # querying tf_to_j_lookup next sees the restored state, not the
    # filtered snapshot our earlier invalidation cached.
    for h in Iterators.reverse(delete_handles)
        undo!(cf, h)
    end
    isempty(delete_handles) || invalidate_tf_to_j_lookup!()

    return return_draws ? (out, raw_prices, raw_demands) : out
end
