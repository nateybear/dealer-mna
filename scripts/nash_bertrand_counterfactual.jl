using DealerMnA
using .Demand
using LoggingExtras
using LinearAlgebra
using Statistics

# Convergence tolerance in std-price units; also the max outer-sweep count.
const NB_TOL = 1e-4
const NB_MAX_SWEEPS = 100
# Damping factor for the fixed-point iterate: p ← λ·p_new + (1−λ)·p_old.
# Below 1 dampens oscillation; 1 is the raw markup mapping.
const NB_DAMPING = 0.7f0

function (@main)(_)
    snakemake!()
    with_logging() do
        model = Demand.DemandModel(;
            JLD2.load_object(snakemake.input["model_frames"])...
        )
        params = JLD2.load_object(snakemake.input["demand_estimates"]).minimizer
        cf = Demand.Counterfactual(
            model, params; decomp_path = snakemake.input["delta_decomp"]
        )
        estimates = JLD2.load_object(snakemake.input["bbl_estimates"])
        price_densities = JLD2.load_object(snakemake.input["price_densities"])
        state_transition = JLD2.load_object(snakemake.input["state_transition"])
        bbl_matrix_obj = JLD2.load_object(snakemake.input["bbl_matrix"])
        index = bbl_matrix_obj.index
        price_scale = Float64(model.price_scale)

        # -----------------------------------------------------------------------------
        # Step 1a: singleton-FE eviction (mirrors bbl_forward_simulation_stratified.jl).
        # sweep_δ! per quarter afterward (AR correction).
        # -----------------------------------------------------------------------------
        fe_df = cf.regout.fe
        identifiable_vehicleids = Set{String}(
            row.vehicleid for row in eachrow(fe_df) if !ismissing(row.fe_vehicleid)
        )
        identifiable_dealers = Set{UUID}(
            row.dealer_uid for row in eachrow(fe_df) if !ismissing(row.fe_dealer_uid)
        )
        vk_to_vehicleid(vk) = string(vk[2], "_", vk[3], "_", Float32(vk[4]))

        n_evicted_singleton = 0
        for (tjf_key, _) in collect(pairs(cf.model.product_lookup.tjf))
            (t_key, j_key, f_key) = tjf_key
            bad_dealer = f_key ∉ identifiable_dealers
            bad_vk = vk_to_vehicleid(j_key) ∉ identifiable_vehicleids
            (bad_dealer || bad_vk) || continue
            h = apply!(
                cf, Demand.ModelDelete(vehicle = j_key, quarter = t_key, dealer_uid = f_key)
            )
            ismissing(h) || (n_evicted_singleton += 1)
        end
        Supply.invalidate_tf_to_j_lookup!()
        for t_sweep in unique(k[1] for k in keys(cf.model.product_lookup.tjf))
            Demand.sweep_δ!(cf, t_sweep)
        end
        Demand.refresh!(cf)
        @info "Singleton-FE eviction complete" n_evicted_singleton

        # -----------------------------------------------------------------------------
        # Step 1b: non-usable VK eviction (matches sim's usable_vks universe).
        # -----------------------------------------------------------------------------
        T = QuarterKey[(y, q) for y in 2017:2019 for q in 1:4]
        max_model_year = maximum(y for (y, _) in T)
        usable_vks = intersect(
            Set(keys(price_densities)),
            Set(keys(state_transition)),
            Set(keys(model.product_lookup.j)),
        )
        filter!(vk -> vk[3] <= max_model_year, usable_vks)

        n_evicted_usable = 0
        for (tjf_key, _) in collect(pairs(cf.model.product_lookup.tjf))
            (t_key, j_key, f_key) = tjf_key
            j_key ∈ usable_vks && continue
            h = apply!(
                cf, Demand.ModelDelete(vehicle = j_key, quarter = t_key, dealer_uid = f_key)
            )
            ismissing(h) || (n_evicted_usable += 1)
        end
        Supply.invalidate_tf_to_j_lookup!()
        for t_sweep in unique(k[1] for k in keys(cf.model.product_lookup.tjf))
            Demand.sweep_δ!(cf, t_sweep)
        end
        Demand.refresh!(cf)
        @info "Non-usable VK eviction complete" n_evicted_usable n_usable_vks = length(usable_vks)

        # -----------------------------------------------------------------------------
        # Snapshot observed prices for every remaining (t, j, f) triple, in a fixed
        # panel_meta ordering that both scenarios' results align to. product_info[.].p
        # holds the ORIGINAL observed price (PriceChange does not overwrite it), so
        # this is a valid pre-tweak snapshot.
        # -----------------------------------------------------------------------------
        panel_meta = Tuple{QuarterKey, VehicleKey, UUID}[]
        p_observed = Float32[]
        for ((t, f), js) in Supply.tf_to_j_lookup(cf)
            for j in js
                idx = cf.model.product_lookup.tjf[(t, j, f)]
                push!(panel_meta, (t, j, f))
                push!(p_observed, cf.model.product_info[idx].p)
            end
        end
        panel_key_to_idx = Dict(pm => i for (i, pm) in enumerate(panel_meta))
        @info "Panel snapshot" n_triples = length(panel_meta) n_tf_pairs = length(
            Supply.tf_to_j_lookup(cf),
        )

        # Read the CURRENT (post-tweak) price for a (t, j, f) triple from
        # model.X_blocks[bi][K+1, k_local]. product_info.p is not refreshed by
        # PriceChange, so we can't use it for the live price.
        K = cf.model.K
        function current_price(t, j, f)
            idx = cf.model.product_lookup.tjf[(t, j, f)]
            bi, k_local = cf.model.tjf_to_block_k[idx]
            return cf.model.X_blocks[bi][K + 1, k_local]
        end

        # -----------------------------------------------------------------------------
        # mc lookup builder for a given scenario NamedTuple (from bbl_estimates).
        # Scenario.w is a length-n_j vector aligned to index.vehicle_cols;
        # scenario.f to index.dealer_cols; scenario.t is a length-4 vector by
        # quarter_of_year. mc(t, j, f) = w[j] + f_fe[f] + t_fe[quarter_of_year(t)].
        # -----------------------------------------------------------------------------
        vk_pos = Dict(vk => i for (i, vk) in enumerate(index.vehicle_cols))
        dealer_pos = Dict(f => i for (i, f) in enumerate(index.dealer_cols))
        function build_mc_lookup(scenario)
            w_vec = scenario.w
            f_vec = scenario.f
            t_vec = scenario.t
            return function (quarter::QuarterKey, vk::VehicleKey, dealer_uid::UUID)
                return Float32(
                    w_vec[vk_pos[vk]] + f_vec[dealer_pos[dealer_uid]] + t_vec[quarter[2]],
                )
            end
        end

        # -----------------------------------------------------------------------------
        # Bertrand-Nash Gauss-Seidel fixed point over all (t, f) pairs. Standard
        # BLP markup mapping p ← mc − (Jᵀ)⁻¹ q, damped by NB_DAMPING. Applied
        # via PriceChange so that when the next (t, f) reads its own q / J, it
        # sees the previously-updated prices (that's the Gauss-Seidel property).
        # -----------------------------------------------------------------------------
        function solve_nash_bertrand!(cf, mc_of)
            tol_std = NB_TOL
            history = Float32[]
            for outer in 1:NB_MAX_SWEEPS
                max_Δ = 0.0f0
                for ((t, f), js) in collect(Supply.tf_to_j_lookup(cf))
                    isempty(js) && continue
                    q = expected_demand(cf, f, t)
                    J = share_matrix(cf, f, t)
                    mc = Float32[mc_of(t, j, f) for j in js]
                    markup = Float32.(J' \ Float64.(q))
                    p_target = mc .- markup
                    for (i, j) in enumerate(js)
                        p_old = current_price(t, j, f)
                        p_new = NB_DAMPING * p_target[i] + (1 - NB_DAMPING) * p_old
                        max_Δ = max(max_Δ, abs(p_new - p_old))
                        apply!(
                            cf,
                            Demand.PriceChange(
                                vehicle = j, quarter = t, dealer_uid = f, price = p_new,
                            ),
                        )
                    end
                end
                push!(history, max_Δ)
                @info "NB outer sweep" outer max_Δ max_Δ_dollars = max_Δ * price_scale
                if max_Δ < tol_std
                    @info "NB converged" outer max_Δ
                    return history, true
                end
            end
            @warn "NB did not converge" max_sweeps = NB_MAX_SWEEPS final_Δ =
                isempty(history) ? NaN32 : history[end]
            return history, false
        end

        # Read current equilibrium (prices + demands) into panel_meta ordering.
        function read_equilibrium(cf)
            p_eq = Vector{Float32}(undef, length(panel_meta))
            q_eq = Vector{Float32}(undef, length(panel_meta))
            for ((t, f), js) in Supply.tf_to_j_lookup(cf)
                q_vec = expected_demand(cf, f, t)
                for (i, j) in enumerate(js)
                    idx = panel_key_to_idx[(t, j, f)]
                    p_eq[idx] = current_price(t, j, f)
                    q_eq[idx] = q_vec[i]
                end
            end
            return p_eq, q_eq
        end

        # Reset cf's prices to observed. Uses PriceChange so δ shifts by
        # β_price·(observed − current); LP polytope invariants are preserved.
        function restore_observed_prices!(cf)
            for (i, (t, j, f)) in enumerate(panel_meta)
                apply!(
                    cf,
                    Demand.PriceChange(
                        vehicle = j, quarter = t, dealer_uid = f, price = p_observed[i],
                    ),
                )
            end
            Demand.refresh!(cf)
        end

        # -----------------------------------------------------------------------------
        # Solve NB for both scenarios; between scenarios restore observed prices so
        # each starts from the same initial point.
        # -----------------------------------------------------------------------------
        function run_scenario(name::Symbol, scenario)
            @info "Solving Nash-Bertrand: $name"
            mc_of = build_mc_lookup(scenario)
            mc_vec = Float32[mc_of(t, j, f) for (t, j, f) in panel_meta]
            history, converged = solve_nash_bertrand!(cf, mc_of)
            p_NB, q_NB = read_equilibrium(cf)
            gap = p_NB .- p_observed
            weights = q_NB ./ max(sum(q_NB), eps())
            avg_gap_std = sum(weights .* gap)
            avg_p_NB_std = sum(weights .* p_NB)
            avg_p_obs_std = sum(weights .* p_observed)
            @info "$name results" avg_gap_std avg_gap_dollars = avg_gap_std * price_scale avg_p_NB_dollars = avg_p_NB_std * price_scale avg_p_obs_dollars = avg_p_obs_std * price_scale mean_gap_frac = mean(gap ./ max.(p_observed, 1.0f-8)) q_gap_25 = quantile(gap, 0.25) q_gap_50 = quantile(gap, 0.5) q_gap_75 = quantile(gap, 0.75) converged
            return (;
                p_observed = copy(p_observed),
                p_NB, q_NB,
                mc = mc_vec,
                panel_meta = copy(panel_meta),
                history, converged,
            )
        end

        best_case_result = run_scenario(:best_case, estimates.best_case)
        restore_observed_prices!(cf)
        worst_case_result = run_scenario(:worst_case, estimates.worst_case)

        JLD2.save_object(
            snakemake.output["counterfactual"],
            (;
                best_case = best_case_result,
                worst_case = worst_case_result,
                price_scale = Float32(price_scale),
                n_triples = length(panel_meta),
            ),
        )
        @info "Wrote Nash-Bertrand counterfactual" path = snakemake.output["counterfactual"] n_triples = length(
            panel_meta,
        )
    end
    return nothing
end
