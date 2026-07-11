#### Residual demand summaries

using Distributions
using DataFrames
using DataStructures

const _PRICE_SEEDS = Ref{Matrix{Float32}}()
const SIM_SIZE = 150

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
    residual_demand(cf, dealer_uid, quarter; return_draws=false, keep=nothing)

Perturbs SIM_SIZE prices for the dealer's assortment at `(quarter, dealer_uid)`
and OLS-fits per-product intercept + own-/cross-price coefficients. Order is
`tf_to_j_lookup`'s (k_local-sorted), so the internal PriceMultiply loop and
`expected_demand` return aligned vectors.

Reads the assortment via `tf_to_j_lookup(cf)` after invalidating the memoize
cache, so tweaks applied by the caller before the call are reflected. Pass
`keep::Set{VehicleKey}` to restrict the perturbed assortment to those VKs;
non-kept products are ModelDelete'd here and restored before returning.
"""
function residual_demand(
    cf::Counterfactual, dealer_uid::UUID, quarter;
    return_draws::Bool = false,
    keep::Union{Nothing, Set{VehicleKey}} = nothing,
)
    invalidate_tf_to_j_lookup!()
    products = collect(keys(cf.model.product_lookup.j))
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
    product_idxs = [findfirst(==(j), products) for j in carried_products]
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

    for (i, j) in enumerate(product_idxs)
        X = copy(prices)
        X[:, Not(i)] .*= @view X[:, i]
        X = hcat(ones(SIM_SIZE), X)
        α = X \ view(demands, :, i)
        out[j, 1] = α[1]
        out[j, product_idxs .+ 1] .= α[2:end]
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
