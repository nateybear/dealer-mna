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
        sales_booster = bootup(sales_bytes, ncols_state + 1)
        price_mean_boosters = Dict{VehicleKey, XGBoost.Booster}()
        price_var_boosters = Dict{VehicleKey, XGBoost.Booster}()
        for (vk, (mean_bytes, var_bytes)) in price_densities
            price_mean_boosters[vk] = bootup(mean_bytes, ncols_state + 1)
            price_var_boosters[vk] = bootup(var_bytes, ncols_state + 1)
        end
        st_int_mean_boosters = Dict{VehicleKey, XGBoost.Booster}()
        st_int_var_boosters = Dict{VehicleKey, XGBoost.Booster}()
        st_own_mean_boosters = Dict{VehicleKey, XGBoost.Booster}()
        st_own_var_boosters = Dict{VehicleKey, XGBoost.Booster}()
        st_cross_mean_boosters = Dict{VehicleKey, XGBoost.Booster}()
        st_cross_var_boosters = Dict{VehicleKey, XGBoost.Booster}()
        for (vk, tup) in state_transition
            im_b, iv_b, om_b, ov_b, cm_b, cv_b = tup
            st_int_mean_boosters[vk] = bootup(im_b, ncols_state)
            st_int_var_boosters[vk] = bootup(iv_b, ncols_state)
            st_own_mean_boosters[vk] = bootup(om_b, ncols_state)
            st_own_var_boosters[vk] = bootup(ov_b, ncols_state)
            if !isempty(cm_b)
                st_cross_mean_boosters[vk] = bootup(cm_b, ncols_state + 1)
                st_cross_var_boosters[vk] = bootup(cv_b, ncols_state + 1)
            end
        end

        # Every VK the simulator can sample must have a full complement of
        # boosters (price mean + var, state transition intercept/own; cross is
        # optional per state_transition_estimation.jl:127). Filter to that
        # intersection so no runtime lookup can miss.
        usable_vks = intersect(
            Set(keys(price_mean_boosters)),
            Set(keys(price_var_boosters)),
            Set(keys(st_int_mean_boosters)),
            Set(keys(st_own_mean_boosters)),
            Set(keys(model.product_lookup.j)),
        )
        @info "Usable VehicleKey universe" n_usable = length(usable_vks) n_dropped = length(
            model.product_lookup.j
        ) - length(usable_vks)

        for (_, vks) in group_to_vks
            filter!(vk -> vk ∈ usable_vks, vks)
        end
        filter!(kv -> !isempty(kv.second), group_to_vks)

        # -----------------------------------------------------------------------------
        # Time loop scaffolding
        # -----------------------------------------------------------------------------
        T::Vector{QuarterKey} = [(y, q) for y in 2017:2019 for q in 1:4]
        t₀ = T[1]

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
        for f in dealers
            obs_assort = VehicleKey[j for j in intermed[(t₀, f)] if j ∈ usable_vks]
            obs_prices = Float32[
                model.product_info[model.product_lookup.tjf[(t₀, j, f)]].p for j in obs_assort
            ]
            outputs[(t₀, f)] = (
                assortment = obs_assort,
                prices = obs_prices,
                q = Vector{Float32}(expected_demand(cf, f, t₀)),
                phi = NaN32,
                iota = iota_prev[f],
                state_expected = Float32[],
                state_actual = state_prev[f],
                perturbation_prices = Matrix{Float32}(undef, 0, 0),
                perturbation_demands = Matrix{Float32}(undef, 0, 0),
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

                # Sales target φ_t = f(state_{t-1}, ι_{t-1})
                x_st = reshape(vcat(s_prev, ι_prev_f32), 1, :)
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
                    p_sale = if iota_prev[f] && haskey(rho_boosters, group)
                        ρ̃ = clamp(
                            logistic(
                                only(XGBoost.predict(rho_boosters[group], dm_state; margin = true))
                            ),
                            1e-8, 1 - 1e-8,
                        )
                        p_base + (1 - p_base) * ρ̃
                    else
                        p_base
                    end
                    @show group p_sale
                    if rand(rng) < p_sale
                        for vk in get(group_to_vks, group, VehicleKey[])
                            push!(new_assortment, vk)
                        end
                    end
                end
                finalize(dm_state)
                assortment_by_dealer[f] = new_assortment

                # Expected state s_t via transition boosters (dealer's belief).
                # Derived from s_prev and new_assortment only — doesn't need cf
                # to be mutated, so we compute it before applying tweaks. This
                # matters because ModelAdd(price=p) below takes the drawn price
                # directly and skips the hausman_fn lookup that would otherwise
                # NaN out for (t, j, f) triples not in training data.
                # Cross-elasticity predictions batched across partners per focal
                # product — one DMatrix per (vk, all partners) instead of one
                # per partner.
                draw_normal(μ, var) = μ + sqrt(max(Float32(var), 0.0f0)) * Float32(randn(rng))
                state_exp_mat = fill(NaN32, J, J + 1)
                dm_prev_row = XGBoost.DMatrix(reshape(s_prev, 1, :))
                for vk in new_assortment
                    j_idx = prod_index[vk]
                    μ_int = Float32(only(XGBoost.predict(st_int_mean_boosters[vk], dm_prev_row)))
                    v_int = Float32(only(XGBoost.predict(st_int_var_boosters[vk], dm_prev_row)))
                    state_exp_mat[j_idx, 1] = draw_normal(μ_int, v_int)
                    μ_own = Float32(only(XGBoost.predict(st_own_mean_boosters[vk], dm_prev_row)))
                    v_own = Float32(only(XGBoost.predict(st_own_var_boosters[vk], dm_prev_row)))
                    state_exp_mat[j_idx, j_idx + 1] = draw_normal(μ_own, v_own)
                    # Cross booster is optional (empty in training for VKs that
                    # never had a co-carried partner) — skip cross terms if missing.
                    haskey(st_cross_mean_boosters, vk) || continue
                    partners = Int[prod_index[vk2] for vk2 in new_assortment if vk2 != vk]
                    isempty(partners) && continue
                    X_cross = Matrix{Float32}(undef, length(partners), ncols_state + 1)
                    for i in eachindex(partners)
                        X_cross[i, 1:ncols_state] .= s_prev
                        X_cross[i, end] = Float32(partners[i])
                    end
                    dm_cross = XGBoost.DMatrix(X_cross)
                    cross_μ = XGBoost.predict(st_cross_mean_boosters[vk], dm_cross)
                    cross_v = XGBoost.predict(st_cross_var_boosters[vk], dm_cross)
                    for (i, k_idx) in enumerate(partners)
                        state_exp_mat[j_idx, k_idx + 1] = draw_normal(cross_μ[i], cross_v[i])
                    end
                    finalize(dm_cross)
                end
                finalize(dm_prev_row)
                state_expected_flat = reshape(state_exp_mat, :)
                state_expected_by_dealer[f] = state_expected_flat

                # Sample prices — price density conditions on (s_t_expected, log φ_t).
                x_price_common = vcat(state_expected_flat, Float32(log_phi))
                dm_price = XGBoost.DMatrix(reshape(x_price_common, 1, :))
                drawn_prices = Vector{Float32}(undef, length(new_assortment))
                for (i, vk) in enumerate(new_assortment)
                    log_p_mean = only(XGBoost.predict(price_mean_boosters[vk], dm_price))
                    log_p_var = only(XGBoost.predict(price_var_boosters[vk], dm_price))
                    log_p_stddev = sqrt(max(Float32(log_p_var), 0.0f0))
                    log_p = log_p_mean + log_p_stddev * Float32(randn(rng))
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
                        apply!(
                            cf,
                            Demand.ModelAdd(
                                vehicle = vk, quarter = t, dealer_uid = f, price = p
                            ),
                        )
                    end
                end
            end

            Demand.sweep_δ!(cf, t)

            # Phase 2: q_t, ι_t, actual state
            for f in dealers
                q_vec = Vector{Float32}(expected_demand(cf, f, t))
                q_tot = sum(q_vec)
                φ_f = phi_by_dealer[f]
                p_iota = logistic((q_tot - φ_f) / τ_sales)
                new_iota = rand(rng) < p_iota

                s_actual, pert_prices, pert_demands = Supply.residual_demand(
                    cf, f, t; return_draws = true
                )

                assort = assortment_by_dealer[f]

                outputs[(t, f)] = (
                    assortment = assort,
                    prices = prices_by_dealer[f],
                    q = q_vec,
                    phi = φ_f,
                    iota = new_iota,
                    state_expected = state_expected_by_dealer[f],
                    state_actual = reshape(s_actual, :),
                    perturbation_prices = pert_prices,
                    perturbation_demands = pert_demands,
                )

                state_prev[f] = reshape(s_actual, :)
                iota_prev[f] = new_iota
            end
        end

        # -----------------------------------------------------------------------------
        # Persist
        # -----------------------------------------------------------------------------
        mkpath(dirname(snakemake.output["sim"]))
        JLD2.save_object(snakemake.output["sim"], outputs)
        @info "wrote" path = snakemake.output["sim"] n_records = length(outputs)
    end
    return nothing
end
