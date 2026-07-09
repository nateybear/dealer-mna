using DealerMnA
using .Demand
using DataFrames

function (@main)(_)
    snakemake!()
    with_logging() do
        model = DemandModel(;
            JLD2.load_object(snakemake.input["model_frames"])...
        )
        params = JLD2.load_object(snakemake.input["demand_estimates"]).minimizer
        cf = Counterfactual(model, params)

        # Build the kept-VehicleKey set from (model, extra) rows in kept_trims,
        # joined against price_table's (make, model, model_year, trim_rank, extra)
        # mapping — same map used in scripts/bbl_forward_simulation.jl.
        kept_pairs = Set(
            (row.model, row.extra) for
            row in eachrow(read_arrow(snakemake.input["kept_trims"]))
        )
        price_table = read_arrow(snakemake.input["price_table"])
        keep = Set{VehicleKey}()
        for row in eachrow(price_table)
            vk = (row.make, row.model, row.model_year, Float32(row.trim_rank))
            haskey(model.product_lookup.j, vk) || continue
            (row.model, row.extra) ∈ kept_pairs || continue
            push!(keep, vk)
        end
        @info "Residual demand universe" n_kept_vks = length(keep) n_all_vks = length(
            model.product_lookup.j
        )

        iter = Supply.tf_to_j_lookup(cf)
        J = length(keys(model.product_lookup.j))
        out = Matrix{Float32}(undef, length(keys(iter)), J * (J + 1))
        for (i, (t, f)) in enumerate(keys(iter))
            @info "Writing dealer" i t f
            out[i, :] = reshape(Supply.residual_demand(cf, f, t; keep = keep), :)
        end

        JLD2.save_object(snakemake.output["residual_demand"], out)
    end
    return nothing
end
