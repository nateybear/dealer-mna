#### Residual demand summaries

using Distributions
using DataFrames
using DataStructures

const _PRICE_SEEDS = Ref{Matrix{Float32}}()
const SIM_SIZE = 150

@memoize function tf_to_j_lookup(model::DemandModel)
    out = OrderedDict{Tuple{QuarterKey, UUID}, Vector{VehicleKey}}()
    (; tjf) = model.product_lookup
    for (t, j, f) in keys(tjf)
        key = (t, f)
        if haskey(out, key)
            push!(out[key], j)
        else
            out[key] = [j]
        end
    end
    return out
end

tf_to_j_lookup(cf::Counterfactual) = tf_to_j_lookup(cf.model)

function seed!(cf::Counterfactual)
    num_models = length(keys(cf.model.product_lookup.j))
    if !isdefined(_PRICE_SEEDS, 1) || Base.size(_PRICE_SEEDS[], 1) != num_models
        D = Uniform(0.85, 1.15)
        _PRICE_SEEDS[] = rand(D, num_models, SIM_SIZE)
    end
end

function residual_demand(
    cf::Counterfactual, dealer_uid::UUID, quarter;
    return_draws::Bool = false,
    keep::Union{Nothing, Set{VehicleKey}} = nothing,
)
    products = collect(keys(cf.model.product_lookup.j))
    out = fill(NaN32, (length(products), length(products) + 1))
    carried_products = tf_to_j_lookup(cf)[(quarter, dealer_uid)]

    # When a `keep` filter is provided, remove non-kept products from the
    # counterfactual state for this (t, f) so `expected_demand` returns a
    # vector aligned with the reduced `carried_products`. Undone before return.
    delete_handles = NamedTuple[]
    if keep !== nothing
        for j in carried_products
            j ∈ keep && continue
            h = apply!(cf, ModelDelete(vehicle = j, quarter = quarter, dealer_uid = dealer_uid))
            ismissing(h) || push!(delete_handles, h)
        end
        carried_products = [j for j in carried_products if j ∈ keep]
    end
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

    # Restore the ModelDeletes applied at entry (LIFO).
    for h in Iterators.reverse(delete_handles)
        undo!(cf, h)
    end

    return return_draws ? (out, raw_prices, raw_demands) : out
end
