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

    # Kept-VK set — same (model, extra) → VehicleKey map that
    # scripts/bbl_forward_simulation_stratified.jl builds from price_table.
    _kept_pairs = Set(
        (row.model, row.extra) for
        row in eachrow(read_arrow("data/kept_trims.arrow"))
    )
    _price_table = read_arrow("data/price_table.arrow")
    keep = Set{VehicleKey}()
    for row in eachrow(_price_table)
        vk = (row.make, row.model, row.model_year, Float32(row.trim_rank))
        haskey(model.product_lookup.j, vk) || continue
        (row.model, row.extra) ∈ _kept_pairs || continue
        push!(keep, vk)
    end

    # Persistent eviction: remove every non-kept (t, j, f) triple from the
    # demand model model-wide before any residual_demand call. This makes the
    # residual fit and every downstream booster reflect the same reduced-
    # product world that BBL forward simulation runs in — no world-A vs world-B
    # mismatch between training data and the sim.
    let n_evicted = 0
        for (tjf_key, _) in collect(pairs(cf.model.product_lookup.tjf))
            (t_key, j_key, f_key) = tjf_key
            j_key ∈ keep && continue
            h = apply!(
                cf, Demand.ModelDelete(vehicle = j_key, quarter = t_key, dealer_uid = f_key)
            )
            ismissing(h) || (n_evicted += 1)
        end
        Supply.invalidate_tf_to_j_lookup!()
        @info "Persistent eviction of non-kept (t,j,f) triples" n_evicted n_kept_vks = length(keep)
    end

    # Re-sweep δ per quarter under the Ackerberg-Rysman correction. Persistent
    # eviction shrinks J_t (live-product count) in every block, and pred_regout
    # is trained with log(J) as a covariate — so the pre-eviction δ values are
    # stale for the new J_t. Without this sweep, expected_demand inside
    # residual_demand reads δ that reflects the old market size, contaminating
    # every downstream booster. ModelDelete's per-tweak δ shift is a linear
    # offset only; only pred_regout does the full AR correction.
    let quarters = unique(k[1] for k in keys(cf.model.product_lookup.tjf))
        for t in quarters
            Demand.sweep_δ!(cf, t)
        end
        Demand.refresh!(cf)
        @info "Swept δ under AR correction" n_quarters = length(quarters)
    end

    # Canonical kept-VK ordering — column layout for the K × (K+1) output
    # matrix and the reference `products` list every downstream stage indexes
    # into. Sorted so the ordering is deterministic across runs.
    kept_vks = sort(collect(keep))
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
    @info "Residual demand" nworkers = nworkers() n_kept_vks = length(keep) K n_tfs = length(
        tf_keys
    ) n_all_vks = length(model.product_lookup.j)
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
