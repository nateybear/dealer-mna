using DealerMnA
using .Demand
using DataFrames
using Statistics: mean, median, quantile
using Random

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

        # Singleton-FE eviction + full-grid sweep_δ, matching residual_demands.jl.
        # Without this, residual_demand on fresh cf would hit missing FE values
        # and the baseline wouldn't match state_matrix_obs.
        let fe_df = cf.regout.fe
            identifiable_vehicleids = Set{String}(
                row.vehicleid for row in eachrow(fe_df) if !ismissing(row.fe_vehicleid)
            )
            identifiable_dealers = Set{UUID}(
                row.dealer_uid for row in eachrow(fe_df) if !ismissing(row.fe_dealer_uid)
            )
            vk_to_vehicleid(vk) = string(vk[2], "_", vk[3], "_", Float32(vk[4]))
            n_evicted = 0
            for (tjf_key, _) in collect(pairs(cf.model.product_lookup.tjf))
                (t_key, j_key, f_key) = tjf_key
                bad_dealer = f_key ∉ identifiable_dealers
                bad_vk = vk_to_vehicleid(j_key) ∉ identifiable_vehicleids
                (bad_dealer || bad_vk) || continue
                h = apply!(
                    cf,
                    Demand.ModelDelete(
                        vehicle = j_key, quarter = t_key, dealer_uid = f_key
                    ),
                )
                ismissing(h) || (n_evicted += 1)
            end
            Supply.invalidate_tf_to_j_lookup!()
            for t_sweep in unique(k[1] for k in keys(cf.model.product_lookup.tjf))
                Demand.sweep_δ!(cf, t_sweep)
            end
            Demand.refresh!(cf)
            @info "Evicted singleton-FE (t,j,f) triples" n_evicted
        end

        rd = JLD2.load_object(snakemake.input["residual_demand"])
        kept_vks = rd.kept_vks
        K = length(kept_vks)
        ncols = K * (K + 1)
        tf_keys = rd.tf_keys

        # Sample (t, f) pairs with at least a few live products (empty rows are
        # uninformative here — residual_demand returns all-NaN).
        rng_sample = MersenneTwister(20261029)
        n_samples = parse(Int, get(snakemake.params, "n_samples", "25"))
        live_tfs = Tuple{QuarterKey, UUID}[]
        intermed = Supply.tf_to_j_lookup(cf)
        for (t, f) in tf_keys
            vks_tf = get(intermed, (t, f), VehicleKey[])
            length(vks_tf) >= 5 || continue
            push!(live_tfs, (t, f))
        end
        shuffle!(rng_sample, live_tfs)
        samples = live_tfs[1:min(n_samples, length(live_tfs))]
        @info "Sampled (t, f) pairs" n_samples = length(samples) pool = length(live_tfs)

        function reseed_price_seeds!(seed_int)
            rng = MersenneTwister(seed_int)
            Supply._PRICE_SEEDS[] =
                rand(rng, Float32, K, Supply.SIM_SIZE) .* 0.3f0 .+ 0.85f0
            return nothing
        end

        # Baseline: deterministic PRICE_SEEDS matching residual_demands.jl.
        TRAIN_SEED = 20260709
        ALT_SEED_A = 20260801
        ALT_SEED_B = 20260902

        # Per-perturbation accumulators across all sampled (t, f) × column.
        kinds = ["rng_reseed_A", "rng_reseed_B", "drop_one_trim", "prices_x1.05"]
        abs_diffs = Dict(k => Float64[] for k in kinds)
        # Split per coef_type: {intercept, own, cross}
        abs_diffs_by_type = Dict(
            (k, ct) => Float64[] for k in kinds for ct in ("intercept", "own", "cross")
        )

        for (i, (t, f)) in enumerate(samples)
            # Baseline with training seed.
            reseed_price_seeds!(TRAIN_SEED)
            s_base = reshape(Supply.residual_demand(cf, f, t; products = kept_vks), :)

            function record_diff!(kind_label, s_pert)
                for c in 1:ncols
                    b = s_base[c]
                    p = s_pert[c]
                    (isnan(b) || isnan(p)) && continue
                    d = abs(Float64(p - b))
                    push!(abs_diffs[kind_label], d)
                    j_focal = ((c - 1) % K) + 1
                    k_col = ((c - 1) ÷ K) + 1
                    ct = if k_col == 1
                        "intercept"
                    elseif k_col - 1 == j_focal
                        "own"
                    else
                        "cross"
                    end
                    push!(abs_diffs_by_type[(kind_label, ct)], d)
                end
                return nothing
            end

            # RNG-reseed sensitivity (fresh _PRICE_SEEDS, same inputs).
            reseed_price_seeds!(ALT_SEED_A)
            s_rng_a = reshape(Supply.residual_demand(cf, f, t; products = kept_vks), :)
            record_diff!("rng_reseed_A", s_rng_a)
            reseed_price_seeds!(ALT_SEED_B)
            s_rng_b = reshape(Supply.residual_demand(cf, f, t; products = kept_vks), :)
            record_diff!("rng_reseed_B", s_rng_b)

            # Reset to training seed for input-perturbation tests.
            reseed_price_seeds!(TRAIN_SEED)

            # Drop one random product from the assortment.
            vks_tf = get(Supply.tf_to_j_lookup(cf), (t, f), VehicleKey[])
            if !isempty(vks_tf)
                vk_drop = vks_tf[rand(rng_sample, 1:length(vks_tf))]
                h = apply!(
                    cf, Demand.ModelDelete(vehicle = vk_drop, quarter = t, dealer_uid = f)
                )
                if !ismissing(h)
                    reseed_price_seeds!(TRAIN_SEED)
                    s_drop = reshape(
                        Supply.residual_demand(cf, f, t; products = kept_vks), :
                    )
                    record_diff!("drop_one_trim", s_drop)
                    undo!(cf, h)
                end
            end

            # Multiply all prices by 1.05.
            h = apply!(cf, Demand.PriceMultiply(; dealer_uid = f, quarter = t, by = 1.05f0))
            if !ismissing(h)
                reseed_price_seeds!(TRAIN_SEED)
                s_mult =
                    reshape(Supply.residual_demand(cf, f, t; products = kept_vks), :)
                record_diff!("prices_x1.05", s_mult)
                undo!(cf, h)
            end

            i % 5 == 0 && @info "Progress" done = i total = length(samples)
        end

        # Summarize.
        out = DataFrame(
            kind = String[], coef_type = String[], n = Int[],
            mean_abs = Float64[], median_abs = Float64[],
            p90_abs = Float64[], p99_abs = Float64[],
        )
        for kind in kinds
            v_all = abs_diffs[kind]
            push!(
                out,
                (
                    kind, "all", length(v_all),
                    isempty(v_all) ? NaN : mean(v_all),
                    isempty(v_all) ? NaN : median(v_all),
                    isempty(v_all) ? NaN : quantile(v_all, 0.9),
                    isempty(v_all) ? NaN : quantile(v_all, 0.99),
                ),
            )
            for ct in ("intercept", "own", "cross")
                v = abs_diffs_by_type[(kind, ct)]
                push!(
                    out,
                    (
                        kind, ct, length(v),
                        isempty(v) ? NaN : mean(v),
                        isempty(v) ? NaN : median(v),
                        isempty(v) ? NaN : quantile(v, 0.9),
                        isempty(v) ? NaN : quantile(v, 0.99),
                    ),
                )
            end
        end
        Arrow.write(snakemake.output["summary"], out)
        @info "Residual-demand sensitivity summary" rows = nrow(out)
    end
    return nothing
end
