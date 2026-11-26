#### BBL forward simulation: policy draws and counterfactual compliance histories

using XGBoost
using Random: Xoshiro
using LogExpFunctions: logistic

# Everything the policy draw reads, built once per sim run.
struct PolicyBoosters
    sales::XGBoost.Booster
    pbase::Dict{Tuple{String, Int64, String}, XGBoost.Booster}
    rho::Dict{Tuple{String, Int64, String}, XGBoost.Booster}
    group_to_vks::Dict{Tuple{String, Int64, String}, Vector{VehicleKey}}
    price::Dict{VehicleKey, Vector{XGBoost.Booster}}
    st_int::Dict{VehicleKey, Vector{XGBoost.Booster}}
    st_own_fo::Dict{VehicleKey, Vector{XGBoost.Booster}}
    st_own_so::Dict{VehicleKey, Vector{XGBoost.Booster}}
    st_cross_fo::Dict{VehicleKey, Vector{XGBoost.Booster}}
    st_cross_int::Dict{VehicleKey, Vector{XGBoost.Booster}}
    products::Vector{VehicleKey}        # kept_vks: product axis of the state matrix
    prod_index::Dict{VehicleKey, Int}
    ncols_state::Int
end

# One stream per (seed, t, f, purpose, key): every history that reaches (t, f) sees the same
# draws, so counterfactual histories differ only through their state and compliance.
keyed_rng(seed, t, f, tag, key...) = Xoshiro(hash((seed, t, f, tag, key...)))

_grid_draw(grid, dm, u) =
    sum(w * only(XGBoost.predict(grid[i], dm; margin = true)) for (w, i) in interp_quantile_grid(u))

"""
    draw_policy(pb, seed, t, f, s_prev, ι_prev) → (; phi, assortment, state_expected, prices)

One draw from dealer f's policy at t given last quarter's state and compliance: the sales
target, the assortment (per compliance group, with the ρ̃ boost when ι_prev), the expected
state from the transition grids, and prices from the price grids on (expected state, log φ).
"""
function draw_policy(
    pb::PolicyBoosters, seed, t, f, s_prev::AbstractVector{Float32}, ι_prev::Bool
)
    J = length(pb.products)
    ncols_state = pb.ncols_state
    grid_draw(grid, dm, tag, key...) =
        _grid_draw(grid, dm, rand(keyed_rng(seed, t, f, tag, key...)))

    # Sales target φ_t = f(state_{t-1}, quarter_of_year(t), ι_{t-1}).
    # Column order matches X_agg in compliance_estimation.jl.
    dm_st = XGBoost.DMatrix(reshape(vcat(s_prev, Float32(t[2]), Float32(ι_prev)), 1, :))
    log_phi = Float32(only(XGBoost.predict(pb.sales, dm_st; margin = true)))
    finalize(dm_st)

    # Sample assortment_t: iterate over compliance groups
    dm_state = XGBoost.DMatrix(reshape(s_prev, 1, :))
    assortment = VehicleKey[]
    for (group, pb_boost) in pb.pbase
        isrelevant_model = let y = group[2]
            if t[1] == y
                true
            elseif t[1] == y - 1
                t[2] == 4
            elseif t[1] == y + 1
                t[2] == 1
            else
                false
            end
        end
        isrelevant_model || continue
        p_base = clamp(
            logistic(only(XGBoost.predict(pb_boost, dm_state; margin = true))), 1e-8, 1 - 1e-8
        )
        u_base, u_rho = rand(keyed_rng(seed, t, f, :assortment, group), 2)
        is_sold = u_base < p_base
        if !is_sold && ι_prev
            ρ̃ = if haskey(pb.rho, group)
                clamp(
                    rho_from_margin(
                        only(XGBoost.predict(pb.rho[group], dm_state; margin = true))
                    ),
                    Float32(RHO_MIN), 1 - 1.0f-8,
                )
            else
                Float32(RHO_MIN)
            end
            is_sold = u_rho < ρ̃
        end
        is_sold && append!(assortment, get(pb.group_to_vks, group, VehicleKey[]))
    end
    finalize(dm_state)

    # Expected state s_t via quantile-grid transition boosters. X layout:
    # [s_prev, assortment_indicator] (+partner on cross), indicator over `products`.
    state_exp_mat = fill(NaN32, J, state_matrix_width(J))
    x_prev_row = zeros(Float32, ncols_state + J)
    x_prev_row[1:ncols_state] .= s_prev
    for vk in assortment
        x_prev_row[ncols_state + pb.prod_index[vk]] = 1.0f0
    end
    dm_prev_row = XGBoost.DMatrix(reshape(x_prev_row, 1, :))
    for vk in assortment
        j_idx = pb.prod_index[vk]
        state_exp_mat[j_idx, intercept_col()] =
            grid_draw(pb.st_int[vk], dm_prev_row, :st_int, vk)
        state_exp_mat[j_idx, firstorder_col(j_idx)] =
            grid_draw(pb.st_own_fo[vk], dm_prev_row, :st_own_fo, vk)
        state_exp_mat[j_idx, quadratic_col(J, j_idx)] =
            grid_draw(pb.st_own_so[vk], dm_prev_row, :st_own_so, vk)
        haskey(pb.st_cross_fo, vk) || continue
        haskey(pb.st_cross_int, vk) || continue
        partners = Int[pb.prod_index[vk2] for vk2 in assortment if vk2 != vk]
        isempty(partners) && continue
        X_cross = Matrix{Float32}(undef, length(partners), ncols_state + J + 1)
        for i in eachindex(partners)
            X_cross[i, 1:(ncols_state + J)] .= x_prev_row
            X_cross[i, end] = Float32(partners[i])
        end
        dm_cross = XGBoost.DMatrix(X_cross)
        for (i, k_idx) in enumerate(partners)
            dm_i = view(dm_cross, i:i, :)
            state_exp_mat[j_idx, firstorder_col(k_idx)] =
                grid_draw(pb.st_cross_fo[vk], dm_i, :st_cross_fo, vk, k_idx)
            state_exp_mat[j_idx, quadratic_col(J, k_idx)] =
                grid_draw(pb.st_cross_int[vk], dm_i, :st_cross_int, vk, k_idx)
        end
        finalize(dm_cross)
    end
    finalize(dm_prev_row)
    state_expected = reshape(state_exp_mat, :)

    # Sample prices — quantile-grid density on (s_t_expected, log φ_t).
    dm_price = XGBoost.DMatrix(reshape(vcat(state_expected, log_phi), 1, :))
    prices = Float32[
        Float32(exp(grid_draw(pb.price[vk], dm_price, :price, vk))) for vk in assortment
    ]
    finalize(dm_price)

    return (; phi = Float32(exp(log_phi)), assortment, state_expected, prices)
end

"""
    apply_policy!(cf, f, t, current, assortment, prices) → handles

Moves f's live products at t from `current` to `assortment` at `prices`: ModelDelete for dropped
products, PriceChange for kept ones, ModelAdd with the drawn price for new ones. ModelAdd takes
the price directly (bypassing hausman_fn, which would NaN on triples outside training) and leaves
a placeholder δ: the spine sweeps the whole quarter once every dealer is applied, history nodes
call `sweep_added!`.
"""
function apply_policy!(cf::Counterfactual, f, t, current, assortment, prices)
    handles = NamedTuple[]
    current_set = Set(current)
    for vk in setdiff(current, assortment)
        h = apply!(cf, ModelDelete(vehicle = vk, quarter = t, dealer_uid = f))
        ismissing(h) || push!(handles, h)
    end
    for (vk, p) in zip(assortment, prices)
        isnan(p) && continue
        tweak = if vk in current_set
            PriceChange(vehicle = vk, quarter = t, dealer_uid = f, price = p)
        else
            ModelAdd(vehicle = vk, quarter = t, dealer_uid = f, price = p, predict_delta = false)
        end
        h = apply!(cf, tweak)
        ismissing(h) || push!(handles, h)
    end
    return handles
end

# δ̂ for just the products a node added, at the block's current J. A full sweep_δ! would also
# rewrite the spine's δ for the quarter and break undo; skipping it drops the log(J) shift of the
# block's other products, ≈ γ·ΔJ/J.
function sweep_added!(cf::Counterfactual, t, handles)
    added = Set{Int32}(h.δ_idx for h in handles if h.kind === :model_add)
    isempty(added) || sweep_δ!(cf, t; only = added)
    return nothing
end

undo_all!(cf::Counterfactual, handles) = foreach(h -> undo!(cf, h), Iterators.reverse(handles))

# Demand for f's whole assortment at t under each coordinated shift of `prices`.
function coordinated_demands(cf::Counterfactual, f, t, prices)
    coord_prices = Matrix{Float32}(undef, length(COORDINATED_SHIFTS), length(prices))
    coord_demands = similar(coord_prices)
    for (i_shift, α) in enumerate(COORDINATED_SHIFTS)
        h = apply!(cf, PriceMultiply(; dealer_uid = f, quarter = t, by = α))
        coord_demands[i_shift, :] .= expected_demand(cf, f, t)
        coord_prices[i_shift, :] .= α .* prices
        undo!(cf, h)
    end
    return coord_prices, coord_demands
end

# q, prices and coordinated-shift demands for f at t, in tf_to_j_lookup (k_local) order.
function realize(cf::Counterfactual, f, t, assortment, prices)
    invalidate_tf_to_j_lookup!()
    canonical = get(tf_to_j_lookup(cf), (t, f), VehicleKey[])
    if isempty(canonical)
        empty_mat = Matrix{Float32}(undef, 0, 0)
        return (;
            assortment = canonical, prices = Float32[], q = Float32[],
            coordinated_prices = empty_mat, coordinated_demands = empty_mat,
        )
    end
    p = prices[[findfirst(==(vk), assortment) for vk in canonical]]
    q = Vector{Float32}(expected_demand(cf, f, t))
    coord_prices, coord_demands = coordinated_demands(cf, f, t, p)
    return (;
        assortment = canonical, prices = p, q,
        coordinated_prices = coord_prices, coordinated_demands = coord_demands,
    )
end

# (b, bits): the history that left the spine at quarter b and complied per `bits` from b on.
# Spine nodes are (t, []).
const NodeKey = Tuple{QuarterKey, Vector{Bool}}

next_quarter((y, q)) = q == 4 ? (y + 1, 1) : (y, q + 1)

# Quarters of the BBL window starting at t0: WINDOW_QUARTERS long, cut off at T_END.
function window_quarters(t0)
    quarters = QuarterKey[t0]
    while length(quarters) < WINDOW_QUARTERS && quarters[end] < T_END
        push!(quarters, next_quarter(quarters[end]))
    end
    return quarters
end

"""
    simulate_histories!(histories, cf, pb, spine, f, seed, t_seed) → nodes

Follows dealer f's counterfactual histories off the simulated spine for every window in
`T_START`, with compliance ι = 1{Q ≥ φ} under whatever prices the history plays:
- one-shot: residual-demand draw r at t0, then the policy. Every draw that flips ι_t0 shares one
  continuation, built only if some draw flips.
- persistent: coordinated shift m at every quarter of the window.

Nodes are built the first time any history reaches them and shared from then on; a node's state
is computed only once a history needs its children. Rivals stay on the spine. Writes
`histories[(t0, f)]` and returns f's nodes.
"""
function simulate_histories!(
    histories, cf::Counterfactual, pb::PolicyBoosters, spine, f, seed, t_seed
)
    nodes = Dict{NodeKey, NamedTuple}()
    states = Dict{NodeKey, Vector{Float32}}()
    quarter_of(key) = foldl((t, _) -> next_quarter(t), key[2]; init = key[1])
    record(key) = isempty(key[2]) ? spine[(key[1], f)] : nodes[key]
    parent(key) = (key[1], key[2][1:(end - 1)])

    function state!(key)
        isempty(key[2]) && return spine[(key[1], f)].state_actual
        haskey(states, key) && return states[key]
        rec = nodes[key]
        # Empty assortment: the dealer sat the quarter out, so the state carries over.
        isempty(rec.assortment) && return states[key] = state!(parent(key))
        t = quarter_of(key)
        handles = apply_policy!(cf, f, t, spine[(t, f)].assortment, rec.assortment, rec.prices)
        sweep_added!(cf, t, handles)
        s = reshape(residual_demand(cf, f, t; products = pb.products), :)
        undo_all!(cf, handles)
        return states[key] = s
    end

    function step!(key, ι)
        t = quarter_of(key)
        child = if isempty(key[2])
            ι == spine[(t, f)].iota ? (next_quarter(t), Bool[]) : (t, [ι])
        else
            (key[1], [key[2]; ι])
        end
        (isempty(child[2]) || haskey(nodes, child)) && return child
        t_child = next_quarter(t)
        draw = draw_policy(pb, seed, t_child, f, state!(key), ι)
        handles = apply_policy!(
            cf, f, t_child, spine[(t_child, f)].assortment, draw.assortment, draw.prices
        )
        sweep_added!(cf, t_child, handles)
        out = realize(cf, f, t_child, draw.assortment, draw.prices)
        undo_all!(cf, handles)
        iota = isempty(out.assortment) ? ι : sum(out.q) ≥ draw.phi
        nodes[child] = (; out..., phi = draw.phi, iota)
        return child
    end

    # ι decided at `key` by a history playing coordinated shift m there. The seed quarter's ι is
    # observed, and a dealer with no products carries its ι forward.
    function shift_iota(key, m)
        rec = record(key)
        (quarter_of(key) == t_seed || isempty(rec.assortment)) && return rec.iota
        return sum(@view rec.coordinated_demands[m, :]) ≥ rec.phi
    end

    for t0 in T_START
        n_ahead = length(window_quarters(t0)) - 1
        root = (t0, Bool[])
        sp = spine[(t0, f)]
        flips = falses(SIM_SIZE)
        if t0 != t_seed && !isempty(sp.assortment)
            flips .= (vec(sum(sp.perturbation_demands; dims = 2)) .≥ sp.phi) .!= sp.iota
        end
        oneshot_path = NodeKey[]
        if any(flips)
            key = step!(root, !sp.iota)
            push!(oneshot_path, key)
            for _ in 2:n_ahead
                key = step!(key, record(key).iota)
                push!(oneshot_path, key)
            end
        end
        shift_paths = map(eachindex(COORDINATED_SHIFTS)) do m
            local key = root
            path = NodeKey[]
            for _ in 1:n_ahead
                key = step!(key, shift_iota(key, m))
                push!(path, key)
            end
            return path
        end
        histories[(t0, f)] = (; oneshot_flips = flips, oneshot_path, shift_paths)
    end
    return nodes
end
