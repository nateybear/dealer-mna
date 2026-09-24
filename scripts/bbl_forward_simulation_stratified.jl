using DealerMnA
using .Demand
using XGBoost
using LoggingExtras
using LinearAlgebra
using LogExpFunctions: logistic
using Statistics: mean
using StatsBase: quantile
using Random
import Base: size  # disambiguate against Cleaner.size (re-exported via DealerMnA)

BLAS.set_num_threads(1)

# Coordinated uniform price multipliers applied to a dealer's whole assortment,
# separate from the per-product `_PRICE_SEEDS` used to fit residual demand.
# These shift ALL prices together → probes substitution to the outside good,
# which per-product independent wiggles absorb into within-dealer cross-substitution.
const COORDINATED_SHIFTS = Float32[0.75, 0.9, 0.95, 0.98, 0.99, 1.01, 1.02, 1.05, 1.1, 1.25]

function (@main)(_)
    snakemake!()
    with_logging() do
        seed = parse(Int, snakemake.wildcards["seed"])
        rng = MersenneTwister(seed)
        @info "BBL forward simulation" seed

        # -----------------------------------------------------------------------------
        # Load
        # -----------------------------------------------------------------------------
        model = Demand.DemandModel(;
            JLD2.load_object(snakemake.input["model_frames"])...
        )
        params = JLD2.load_object(snakemake.input["demand_estimates"]).minimizer
        cf = Demand.Counterfactual(
            model, params; decomp_path = snakemake.input["delta_decomp"]
        )
        compliance_fits = JLD2.load_object(snakemake.input["compliance_fits"])
        sales_bytes, τ_sales = JLD2.load_object(snakemake.input["sales_targets"])
        price_densities = JLD2.load_object(snakemake.input["price_densities"])
        state_transition = JLD2.load_object(snakemake.input["state_transition"])
        compliance_arrow = read_arrow(snakemake.input["compliance"])
        state_matrix_obs = JLD2.load_object(snakemake.input["residual_demand"])
        price_table = read_arrow(snakemake.input["price_table"])

        # -----------------------------------------------------------------------------
        # Evict products whose dealer_uid or vehicleid has no identifiable FE in
        # regout. FixedEffectModels writes `missing` to fe_* for iteratively-
        # dropped singleton groups, and pred_regout (src/demand/dense.jl:738-741)
        # chokes on missing FE values. sweep_δ! iterates every product in the
        # block, so filtering BBL-side inputs isn't enough — we have to remove
        # them from the DemandModel. ModelDelete is persistent (no undo):
        # cf.model keeps the filtered state for the whole sim.
        # -----------------------------------------------------------------------------
        fe_df = cf.regout.fe
        identifiable_vehicleids = Set{String}(
            row.vehicleid for row in eachrow(fe_df) if !ismissing(row.fe_vehicleid)
        )
        identifiable_dealers = Set{UUID}(
            row.dealer_uid for row in eachrow(fe_df) if !ismissing(row.fe_dealer_uid)
        )
        vk_to_vehicleid(vk) = string(vk[2], "_", vk[3], "_", Float32(vk[4]))

        n_evicted = 0
        n_evicted_dealer = 0
        n_evicted_vehicleid = 0
        for (tjf_key, _) in collect(pairs(cf.model.product_lookup.tjf))
            (t_key, j_key, f_key) = tjf_key
            bad_dealer = f_key ∉ identifiable_dealers
            bad_vk = vk_to_vehicleid(j_key) ∉ identifiable_vehicleids
            (bad_dealer || bad_vk) || continue
            h = apply!(
                cf, Demand.ModelDelete(vehicle = j_key, quarter = t_key, dealer_uid = f_key)
            )
            ismissing(h) && continue
            n_evicted += 1
            bad_dealer && (n_evicted_dealer += 1)
            bad_vk && (n_evicted_vehicleid += 1)
        end
        Supply.invalidate_tf_to_j_lookup!()
        @info "Evicted singleton-FE tuples from DemandModel" n_evicted n_evicted_dealer n_evicted_vehicleid

        # -----------------------------------------------------------------------------
        # Indices and lookups
        # -----------------------------------------------------------------------------
        intermed = Supply.tf_to_j_lookup(model)
        state_lookup = Dict(k => i for (i, k) in enumerate(keys(intermed)))
        products = collect(keys(model.product_lookup.j))
        J = length(products)
        prod_index = Dict(p => i for (i, p) in enumerate(products))
        ncols_state = size(state_matrix_obs, 2)

        # VehicleKey ↔ compliance group (model, model_year, extra). price_table
        # is the canonical source of the extra ↔ trim_rank mapping — the
        # demand model's VehicleKeys and the compliance groups both flow from
        # it, so joining on trim_rank is exact.
        vk_to_group = Dict{VehicleKey, Tuple{String, Int64, String}}()
        group_to_vks = Dict{Tuple{String, Int64, String}, Vector{VehicleKey}}()
        for row in eachrow(price_table)
            vk = (row.make, row.model, row.model_year, Float32(row.trim_rank))
            haskey(model.product_lookup.j, vk) || continue
            group = (row.model, row.model_year, row.extra)
            vk_to_group[vk] = group
            push!(get!(group_to_vks, group, VehicleKey[]), vk)
        end
        @info "VehicleKey ↔ compliance-group map" n_vks = length(vk_to_group) n_groups = length(
            group_to_vks
        )

        # -----------------------------------------------------------------------------
        # Deserialize boosters up front. Each Booster is stateful — allocate one
        # per key per booster type and reuse throughout the sim.
        # -----------------------------------------------------------------------------
        function bootup(bytes, ncols)
            dummy = XGBoost.DMatrix(zeros(Float32, 1, ncols))
            b = XGBoost.Booster(dummy)
            XGBoost.load!(b, bytes)
            finalize(dummy)
            return b
        end

        pbase_boosters = Dict{Tuple{String, Int64, String}, XGBoost.Booster}()
        rho_boosters = Dict{Tuple{String, Int64, String}, XGBoost.Booster}()
        for (key, (pbase_bytes, rho_bytes)) in compliance_fits
            pbase_boosters[key] = bootup(pbase_bytes, ncols_state)
            if !isempty(rho_bytes)
                rho_boosters[key] = bootup(rho_bytes, ncols_state)
            end
        end
        # +2 = quarter_of_year(t) + ι_lag; must match X_agg column count in
        # scripts/compliance_estimation.jl where the booster is trained.
        sales_booster = bootup(sales_bytes, ncols_state + 2)
        # Quantile-grid densities: one Vector{XGBoost.Booster} per VK per
        # target, aligned to Supply.QUANTILE_LEVELS. If any quantile in a grid
        # is empty (best_round == 0 fits at training time, can't be
        # serialized), we treat the whole grid as unusable and drop the VK
        # from that family — the usable_vks intersection below filters
        # accordingly. Base-score-only stub boosters would predict 0.5 for
        # every quantile, giving nonsense draws.
        Q = length(Supply.QUANTILE_LEVELS)
        function bootup_grid(byte_vecs::Vector{Vector{UInt8}}, ncols::Int)
            all(!isempty, byte_vecs) || return nothing
            grid = Vector{XGBoost.Booster}(undef, length(byte_vecs))
            for (i, b) in enumerate(byte_vecs)
                grid[i] = bootup(b, ncols)
            end
            return grid
        end
        price_boosters = Dict{VehicleKey, Vector{XGBoost.Booster}}()
        for (vk, byte_grid) in price_densities
            grid = bootup_grid(byte_grid, ncols_state + 1)
            grid === nothing || (price_boosters[vk] = grid)
        end
        st_int_boosters = Dict{VehicleKey, Vector{XGBoost.Booster}}()
        st_own_boosters = Dict{VehicleKey, Vector{XGBoost.Booster}}()
        st_cross_boosters = Dict{VehicleKey, Vector{XGBoost.Booster}}()
        for (vk, (int_grid, own_grid, cross_grid)) in state_transition
            int_g = bootup_grid(int_grid, ncols_state)
            int_g === nothing || (st_int_boosters[vk] = int_g)
            own_g = bootup_grid(own_grid, ncols_state)
            own_g === nothing || (st_own_boosters[vk] = own_g)
            # Cross grid is optional (all-empty when n_cross < 20 at fit time,
            # or partially-empty when some quantile fits failed): only store
            # if fully populated.
            cross_g = bootup_grid(cross_grid, ncols_state + 1)
            cross_g === nothing || (st_cross_boosters[vk] = cross_g)
        end

        # -----------------------------------------------------------------------------
        # Time loop scaffolding — declared early so `max_model_year` below can
        # cap the VK universe at the last simulated year.
        # -----------------------------------------------------------------------------
        T::Vector{QuarterKey} = [(y, q) for y in 2017:2019 for q in 1:4]
        t₀ = T[1]
        max_model_year = maximum(y for (y, _) in T)

        # Every VK the simulator can sample must have a full complement of
        # boosters (price mean + var, state transition intercept/own; cross is
        # optional per state_transition_estimation.jl:127). Filter to that
        # intersection so no runtime lookup can miss.
        #
        # Also cap model_year at the last simulated year — MY (max_year+1) VKs
        # would be in-season at Q4 of the last year but their vehicleid isn't
        # in regout.fe (2020 data is incomplete), so sweep_δ! → pred_regout
        # would return `missing` and blow up on Float64 conversion.
        usable_vks = intersect(
            Set(keys(price_boosters)),
            Set(keys(st_int_boosters)),
            Set(keys(st_own_boosters)),
            Set(keys(model.product_lookup.j)),
        )
        filter!(vk -> vk[3] <= max_model_year, usable_vks)
        @info "Usable VehicleKey universe" n_usable = length(usable_vks) n_dropped =
            length(
                model.product_lookup.j
            ) - length(usable_vks) max_model_year

        for (_, vks) in group_to_vks
            filter!(vk -> vk ∈ usable_vks, vks)
        end
        filter!(kv -> !isempty(kv.second), group_to_vks)

        # Drop boosters whose group has no usable VKs: otherwise a Bernoulli
        # fire in the assortment-draw loop hits `get(group_to_vks, group, [])`
        # and adds nothing — a wasted fire that inflates the empty-assortment
        # rate proportional to the fraction of dead groups.
        filter!(kv -> haskey(group_to_vks, kv.first), pbase_boosters)
        filter!(kv -> haskey(group_to_vks, kv.first), rho_boosters)

        # Persistent eviction of non-usable products so cf.model's live universe
        # matches the sim's sampling universe. Without this, unusable-but-FE-
        # identifiable products (in `intermed` but not `usable_vks`) survive the
        # tweak-application block below — the `obs_at_t` filter would skip them,
        # so they'd never be ModelDelete'd. They then leak into
        # `expected_demand`, `residual_demand`, and `sweep_δ!`, producing q /
        # state_actual / perturbation_prices sized for the union rather than for
        # `new_assortment` alone.
        n_evicted_usable = 0
        for (tjf_key, _) in collect(pairs(cf.model.product_lookup.tjf))
            (t_key, j_key, f_key) = tjf_key
            j_key ∈ usable_vks && continue
            h = apply!(
                cf, Demand.ModelDelete(vehicle = j_key, quarter = t_key, dealer_uid = f_key)
            )
            ismissing(h) && continue
            n_evicted_usable += 1
        end
        Supply.invalidate_tf_to_j_lookup!()
        intermed = Supply.tf_to_j_lookup(model)
        @info "Evicted non-usable VKs from DemandModel" n_evicted_usable

        dealers = sort(UUID[f for ((t, f), _) in state_lookup if t == t₀])
        @info "Seed period dealer set" n_dealers = length(dealers)

        # p_compliant at t₀ from compliance.arrow — seed for ι_lag.
        p_compliant_t0 = Dict{UUID, Float32}()
        for row in eachrow(compliance_arrow)
            if (Dates.year(row.t), Dates.quarterofyear(row.t)) == t₀
                p_compliant_t0[row.dealer_uid] = Float32(row.p_compliant)
            end
        end

        state_prev = Dict{UUID, Vector{Float32}}()
        iota_prev = Dict{UUID, Bool}()
        for f in dealers
            state_prev[f] = Vector{Float32}(view(state_matrix_obs, state_lookup[(t₀, f)], :))
            iota_prev[f] = rand(rng) < get(p_compliant_t0, f, 0.5f0)
        end

        outputs = Dict{Tuple{QuarterKey, UUID}, NamedTuple}()

        # Seed period record: observed everything, no simulated draws yet.
        # Filter obs to usable_vks so seed & simulated periods are apples-to-apples.
        # Perturbation / coordinated matrices are filled with observed values
        # broadcast across the scenario dim: BBL perturbations only apply for
        # fwd≥1, so at fwd=0 the perturbed lookup must return the same p,q as
        # the observed lookup — otherwise obs−pert leaves a spurious
        # "shutdown-at-seed" moment (β·q·(p−mc)) contaminating every scenario.
        for f in dealers
            obs_assort = VehicleKey[j for j in intermed[(t₀, f)] if j ∈ usable_vks]
            obs_prices = Float32[
                model.product_info[model.product_lookup.tjf[(t₀, j, f)]].p for j in obs_assort
            ]
            obs_q = Vector{Float32}(expected_demand(cf, f, t₀))
            outputs[(t₀, f)] = (
                assortment = obs_assort,
                prices = obs_prices,
                q = obs_q,
                phi = NaN32,
                iota = iota_prev[f],
                state_expected = Float32[],
                state_actual = state_prev[f],
                perturbation_prices = repeat(reshape(obs_prices, 1, :), Supply.SIM_SIZE, 1),
                perturbation_demands = repeat(reshape(obs_q, 1, :), Supply.SIM_SIZE, 1),
                coordinated_shifts = COORDINATED_SHIFTS,
                coordinated_prices = repeat(
                    reshape(obs_prices, 1, :), length(COORDINATED_SHIFTS), 1
                ),
                coordinated_demands = repeat(
                    reshape(obs_q, 1, :), length(COORDINATED_SHIFTS), 1
                ),
            )
        end

        # -----------------------------------------------------------------------------
        # Simulation loop Q2 2017 → Q4 2019
        # -----------------------------------------------------------------------------
        for t in T[2:end]
            @info "Simulating period" t

            phi_by_dealer = Dict{UUID, Float32}()
            log_phi_by_dealer = Dict{UUID, Float32}()
            state_expected_by_dealer = Dict{UUID, Vector{Float32}}()
            assortment_by_dealer = Dict{UUID, Vector{VehicleKey}}()
            prices_by_dealer = Dict{UUID, Vector{Float32}}()  # parallel to assortment

            for f in dealers
                s_prev = state_prev[f]
                ι_prev_f32 = Float32(iota_prev[f])

                # Sales target φ_t = f(state_{t-1}, quarter_of_year(t), ι_{t-1}).
                # Column order matches X_agg in compliance_estimation.jl.
                x_st = reshape(vcat(s_prev, Float32(t[2]), ι_prev_f32), 1, :)
                dm_st = XGBoost.DMatrix(x_st)
                log_phi = only(XGBoost.predict(sales_booster, dm_st; margin = true))
                finalize(dm_st)
                log_phi_by_dealer[f] = Float32(log_phi)
                phi_by_dealer[f] = Float32(exp(log_phi))

                # Sample assortment_t: iterate over compliance groups
                dm_state = XGBoost.DMatrix(reshape(s_prev, 1, :))
                new_assortment = VehicleKey[]
                for (group, pb_boost) in pbase_boosters
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
                        logistic(only(XGBoost.predict(pb_boost, dm_state; margin = true))),
                        1e-8, 1 - 1e-8,
                    )
                    is_sold = if iota_prev[f]
                        ρ̃ = if haskey(rho_boosters, group)
                            clamp(
                                Supply.rho_from_margin(
                                    only(
                                        XGBoost.predict(
                                            rho_boosters[group], dm_state; margin = true
                                        ),
                                    ),
                                ),
                                Float32(Supply.RHO_MIN), 1 - 1.0f-8,
                            )
                        else
                            Float32(Supply.RHO_MIN)
                        end
                        (rand(rng) < p_base) || (rand(rng) < ρ̃)
                    else
                        rand(rng) < p_base
                    end
                    if is_sold
                        for vk in get(group_to_vks, group, VehicleKey[])
                            push!(new_assortment, vk)
                        end
                    end
                end
                finalize(dm_state)
                assortment_by_dealer[f] = new_assortment

                # Expected state s_t via quantile-grid transition boosters.
                # Draws are inverse-CDF from the 9-knot quantile grid (with
                # sort! per row to fix quantile crossing before interpolation).
                # Same overall structure as the mean+var Normal version but
                # each single-row prediction fans out to Q boosters.
                function draw_scalar_grid(grid, dm)
                    weights_idxs = Supply.sample_from_quantile_grid(rng)
                    return sum(
                        w * only(XGBoost.predict(grid[i], dm; margin = true)) for
                        (w, i) in weights_idxs
                    )
                end
                state_exp_mat = fill(NaN32, J, J + 1)
                dm_prev_row = XGBoost.DMatrix(reshape(s_prev, 1, :))
                for vk in new_assortment
                    j_idx = prod_index[vk]
                    state_exp_mat[j_idx, 1] = draw_scalar_grid(st_int_boosters[vk], dm_prev_row)
                    state_exp_mat[j_idx, j_idx + 1] = draw_scalar_grid(
                        st_own_boosters[vk], dm_prev_row
                    )
                    # Cross grid is optional (empty for VKs that never had a
                    # co-carried partner in training) — skip cross terms.
                    haskey(st_cross_boosters, vk) || continue
                    partners = Int[prod_index[vk2] for vk2 in new_assortment if vk2 != vk]
                    isempty(partners) && continue
                    X_cross = Matrix{Float32}(undef, length(partners), ncols_state + 1)
                    for i in eachindex(partners)
                        X_cross[i, 1:ncols_state] .= s_prev
                        X_cross[i, end] = Float32(partners[i])
                    end
                    dm_cross = XGBoost.DMatrix(X_cross)
                    # Batch quantile predictions across all partners: Q calls
                    # total (one per quantile booster) instead of Q per partner.
                    cross_grid = st_cross_boosters[vk]
                    for (i, k_idx) in enumerate(partners)
                        state_exp_mat[j_idx, k_idx + 1] = draw_scalar_grid(
                            cross_grid, view(dm_cross, i:i, :)
                        )
                    end
                    finalize(dm_cross)
                end
                finalize(dm_prev_row)
                state_expected_flat = reshape(state_exp_mat, :)
                state_expected_by_dealer[f] = state_expected_flat

                # Sample prices — quantile-grid density on (s_t_expected, log φ_t).
                x_price_common = vcat(state_expected_flat, Float32(log_phi))
                dm_price = XGBoost.DMatrix(reshape(x_price_common, 1, :))
                drawn_prices = Vector{Float32}(undef, length(new_assortment))
                for (i, vk) in enumerate(new_assortment)
                    log_p = draw_scalar_grid(price_boosters[vk], dm_price)
                    drawn_prices[i] = Float32(exp(log_p))
                end
                finalize(dm_price)
                prices_by_dealer[f] = drawn_prices

                # Apply diffs against OBSERVED assortment at (t, f). ModelAdd
                # takes the drawn price directly (bypassing hausman_fn), which
                # matters because (t, j, f) triples that ModelAdd creates
                # aren't in training so the default hausman lookup would NaN
                # out. Products present in obs get PriceChange; new products
                # get ModelAdd(price=p); removed products get ModelDelete.
                obs_at_t = VehicleKey[
                    vk for vk in get(intermed, (t, f), VehicleKey[]) if vk ∈ usable_vks
                ]
                obs_set = Set(obs_at_t)
                for vk in setdiff(obs_at_t, new_assortment)
                    apply!(cf, Demand.ModelDelete(vehicle = vk, quarter = t, dealer_uid = f))
                end
                for (i, vk) in enumerate(new_assortment)
                    isnan(drawn_prices[i]) && continue
                    p = drawn_prices[i]
                    if vk in obs_set
                        apply!(
                            cf,
                            Demand.PriceChange(
                                vehicle = vk, quarter = t, dealer_uid = f, price = p
                            ),
                        )
                    else
                        # `predict_delta=false` skips pred_regout — sweep_δ!
                        # below overwrites the sentinel with the real δ̂.
                        apply!(
                            cf,
                            Demand.ModelAdd(
                                vehicle = vk, quarter = t, dealer_uid = f,
                                price = p, predict_delta = false,
                            ),
                        )
                    end
                end
            end

            Demand.sweep_δ!(cf, t)

            # Phase 2: q_t, ι_t, actual state
            for f in dealers
                # Dealer drew an empty assortment (every compliance-group
                # Bernoulli fired 0). Treat as "sat this quarter out": carry
                # state_prev / iota_prev forward, record an empty-assortment
                # row, don't call residual_demand (which would return NaN and
                # poison next period's features).
                if isempty(assortment_by_dealer[f])
                    outputs[(t, f)] = (
                        assortment = VehicleKey[],
                        prices = Float32[],
                        q = Float32[],
                        phi = phi_by_dealer[f],
                        iota = iota_prev[f],
                        state_expected = state_expected_by_dealer[f],
                        state_actual = state_prev[f],
                        perturbation_prices = Matrix{Float32}(undef, 0, 0),
                        perturbation_demands = Matrix{Float32}(undef, 0, 0),
                        coordinated_shifts = COORDINATED_SHIFTS,
                        coordinated_prices = Matrix{Float32}(undef, 0, 0),
                        coordinated_demands = Matrix{Float32}(undef, 0, 0),
                    )
                    continue
                end

                q_vec = Vector{Float32}(expected_demand(cf, f, t))
                q_tot = sum(q_vec)
                φ_f = phi_by_dealer[f]
                p_iota = logistic((q_tot - φ_f) / τ_sales)
                new_iota = rand(rng) < p_iota

                s_actual, pert_prices, pert_demands = Supply.residual_demand(
                    cf, f, t; return_draws = true
                )

                # `assortment_by_dealer[f]` and `prices_by_dealer[f]` are in
                # `pbase_boosters`-iteration order; `q_vec` and the residual-
                # demand outputs are in `tf_to_j_lookup(cf)` (k_local) order.
                # Reorder the former to match the latter so all five arrays
                # share one canonical column axis on disk.
                canonical = get(Supply.tf_to_j_lookup(cf), (t, f), VehicleKey[])
                perm = [findfirst(==(vk), assortment_by_dealer[f]) for vk in canonical]
                obs_prices = prices_by_dealer[f][perm]

                # Coordinated wide shifts: scale ALL prices for this (dealer,
                # quarter) by α, read structural expected_demand under the shift.
                # `expected_demand` returns in `tf_to_j_lookup(cf)` order, matching
                # `canonical` / `obs_prices`.
                coord_prices = Matrix{Float32}(undef, length(COORDINATED_SHIFTS), length(canonical))
                coord_demands = Matrix{Float32}(
                    undef, length(COORDINATED_SHIFTS), length(canonical)
                )
                for (i_shift, α) in enumerate(COORDINATED_SHIFTS)
                    h = apply!(cf, Demand.PriceMultiply(; dealer_uid = f, quarter = t, by = α))
                    coord_demands[i_shift, :] .= expected_demand(cf, f, t)
                    coord_prices[i_shift, :] .= α .* obs_prices
                    undo!(cf, h)
                end

                outputs[(t, f)] = (
                    assortment = canonical,
                    prices = obs_prices,
                    q = q_vec,
                    phi = φ_f,
                    iota = new_iota,
                    state_expected = state_expected_by_dealer[f],
                    state_actual = reshape(s_actual, :),
                    perturbation_prices = pert_prices,
                    perturbation_demands = pert_demands,
                    coordinated_shifts = COORDINATED_SHIFTS,
                    coordinated_prices = coord_prices,
                    coordinated_demands = coord_demands,
                )

                state_prev[f] = reshape(s_actual, :)
                iota_prev[f] = new_iota
            end
        end

        # -----------------------------------------------------------------------------
        # Persist
        # -----------------------------------------------------------------------------
        n_empty = count(o -> isempty(o.assortment), values(outputs))
        @info "Empty-assortment dealer-quarters" n_empty n_total = length(outputs) frac =
            n_empty / length(outputs)

        mkpath(dirname(snakemake.output["sim"]))
        JLD2.save_object(snakemake.output["sim"], outputs)
        @info "wrote" path = snakemake.output["sim"] n_records = length(outputs)
    end
    return nothing
end
