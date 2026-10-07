using Distributed

@everywhere begin
    using DealerMnA
    using .Demand
    using DataFrames
    using LoggingExtras
    using Random

    model = DemandModel(; JLD2.load_object("data/model_frames.jld2")...)
    params = JLD2.load_object("data/demand_estimates.jld2").minimizer
    cf = Counterfactual(model, params)

    # kept_trims restriction is applied upstream in load_model_frames —
    # every VK in model.product_lookup.j already cleared the cutoff, so no
    # kept-trims eviction is needed here. Only singleton-FE (t,j,f) triples
    # still need pruning, since FixedEffectModels writes `missing` to
    # fe_dealer_uid / fe_vehicleid for iteratively-dropped singleton groups
    # and sweep_δ! would then throw `MethodError: Cannot convert Missing to Float64`.
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
                cf, Demand.ModelDelete(vehicle = j_key, quarter = t_key, dealer_uid = f_key)
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

    # Canonical kept-VK ordering — column layout for the K × (K+1) output
    # matrix and the reference `products` list every downstream stage indexes
    # into. Sorted so the ordering is deterministic across runs.
    kept_vks = sort(collect(keys(cf.model.product_lookup.j)))
    K = length(kept_vks)

    # Pre-seed Supply's price-perturbation matrix from a deterministic RNG so
    # a parallel run matches a single-process run byte-for-byte. Sized to K so
    # every row is used — residual_demand indexes 1:K under the kept ordering.
    let rng = MersenneTwister(20260709)
        Supply._PRICE_SEEDS[] =
            rand(rng, Float32, K, Supply.SIM_SIZE) .* 0.3f0 .+ 0.85f0
    end

    tf_keys = collect(keys(Supply.tf_to_j_lookup(cf)))

    function compute_row(i::Int)
        t, f = tf_keys[i]
        return reshape(Supply.residual_demand(cf, f, t; products = kept_vks), :)
    end
end

function (@main)(_)
    @info "Residual demand" nworkers = nworkers() K n_tfs = length(tf_keys) n_all_vks = length(
        model.product_lookup.j
    )
    out = Matrix{Float32}(undef, length(tf_keys), K * (K + 1))
    fitted = pmap(compute_row, 1:length(tf_keys))
    for (i, row) in enumerate(fitted)
        out[i, :] .= row
    end
    # Bundle matrix with its canonical kept-VK ordering AND its (t, f) row
    # ordering. Both are baked into the matrix layout — every consumer of
    # data/residual_demand.jld2 unpacks all three:
    #   - kept_vks:  column-block reference product list (K products → K×(K+1))
    #   - tf_keys:   row ordering; the un-evicted model's tf_to_j_lookup can
    #                have extra (t, f) pairs that dropped here, so consumers
    #                must key off this list, not re-derive from `model`.
    JLD2.save_object(
        "data/residual_demand.jld2",
        (matrix = out, kept_vks = kept_vks, tf_keys = tf_keys),
    )
    return nothing
end
