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

"""
    residual_demand(cf, dealer_uid, quarter, carried_products; return_draws=false)

Compute the (J × (J+1)) residual-demand row for `(quarter, dealer_uid)` using
exactly `carried_products` as the assortment to perturb. The caller is
responsible for picking a VK set that reflects the *current* counterfactual
state — `tf_to_j_lookup` is memoized on the model and does not track
ModelAdd/ModelDelete tweaks, so relying on it silently gets stale.

Each VK in `carried_products` must have a live (quarter, vk, dealer_uid)
triple in `cf.model.tjf_to_block_k`; otherwise the internal PriceMultiply
call returns `missing` and this function errors.
"""
function residual_demand(
    cf::Counterfactual, dealer_uid::UUID, quarter,
    carried_products::AbstractVector{VehicleKey};
    return_draws::Bool = false,
)
    products = collect(keys(cf.model.product_lookup.j))
    out = fill(NaN32, (length(products), length(products) + 1))
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

    return return_draws ? (out, raw_prices, raw_demands) : out
end
