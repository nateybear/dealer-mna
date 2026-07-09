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

    # Kept-VK set for Supply.residual_demand's `keep` filter — same
    # (model, extra) → VehicleKey map that scripts/bbl_forward_simulation.jl
    # builds from price_table.
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

    # Pre-seed Supply's price-perturbation matrix from a deterministic RNG so
    # a parallel run matches a single-process run byte-for-byte. Otherwise
    # `Supply.seed!(cf)` fires lazily on each worker and each draws its own
    # matrix, making the residual_demand output depend on the worker layout.
    let rng = MersenneTwister(20260709), J = length(keys(model.product_lookup.j))
        Supply._PRICE_SEEDS[] =
            rand(rng, Float32, J, Supply.SIM_SIZE) .* 0.3f0 .+ 0.85f0
    end

    tf_keys = collect(keys(Supply.tf_to_j_lookup(cf)))

    function compute_row(i::Int)
        t, f = tf_keys[i]
        return reshape(Supply.residual_demand(cf, f, t; keep = keep), :)
    end
end

function (@main)(_)
    @info "Residual demand" nworkers = nworkers() n_kept_vks = length(keep) n_tfs = length(
        tf_keys
    ) n_all_vks = length(model.product_lookup.j)
    J = length(keys(model.product_lookup.j))
    out = Matrix{Float32}(undef, length(tf_keys), J * (J + 1))
    fitted = pmap(compute_row, 1:length(tf_keys))
    for (i, row) in enumerate(fitted)
        out[i, :] .= row
    end
    JLD2.save_object("data/residual_demand.jld2", out)
    return nothing
end
