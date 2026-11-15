using DealerMnA
using StatsBase: mean, std, weights
using LaTeXTabulars
using Printf

# Quarters a model year is on sale: Q4 of Y − 1 through Q1 of Y + 1, the window fords.jl keeps.
sale_quarters(y) = [Date(y - 1, 10, 1) + Quarter(k) for k in 0:5]

function (@main)(_)
    snakemake!()
    with_logging() do
        kept = Set(
            (r.model, r.model_year, r.extra) for
            r in eachrow(read_arrow(snakemake.input["kept_trims"]))
        )
        fords = subset(
            read_arrow(snakemake.input["fords"]), :t => ByRow(t -> Date(2017) <= t < Date(2020))
        )
        panel_quarters = sort(unique(fords.t))
        active_dealers = Dict(
            r.t => r.n for r in eachrow(
                combine(groupby(unique(select(fords, :dealer_uid, :t)), :t), nrow => :n)
            )
        )
        sales = subset(
            fords,
            [:model, :model_year, :extra] => ByRow((m, y, e) -> (m, y, e) ∈ kept),
        )

        # A trim is (model, extra) pooled over its kept model years; availability is the share of
        # active dealer-quarters selling it, over the quarters any of those model years is on sale.
        trims = combine(groupby(sales, [:model, :extra])) do g
            window = intersect(union(sale_quarters.(unique(g.model_year))...), panel_quarters)
            n_selling = nrow(unique(select(g, :dealer_uid, :t)))
            return (;
                n_sales = nrow(g),
                availability = n_selling / sum(active_dealers[t] for t in window),
            )
        end
        models = combine(
            groupby(trims, :model),
            nrow => :n_trims,
            :n_sales => sum => :n_sales,
            [:availability, :n_sales] => ((a, w) -> mean(a, weights(w))) => :mean_avail,
            [:availability, :n_sales] =>
                ((a, w) -> std(a, weights(w); corrected = false)) => :std_avail,
        )
        sort!(models, :model)
        @info "Trim availability" models

        rows = [
            [
                r.model, string(r.n_trims), string(r.n_sales),
                @sprintf("%.2f", r.mean_avail), @sprintf("%.2f", r.std_avail),
            ] for r in eachrow(models)
        ]
        open(snakemake.output[1], "w") do io
            return latex_tabular(
                io,
                Tabular("lrrcc"),
                [
                    Rule(:top),
                    ["Model", "No. Trims", "Total Sales", "Mean Availability", "SD Availability"],
                    Rule(:mid),
                    rows...,
                    Rule(:bottom),
                ],
            )
        end
    end
    return nothing
end
