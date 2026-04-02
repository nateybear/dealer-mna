using DealerMnA
using Plots
using StatsBase

snakemake!()

with_logging() do
    price_table = Registrations.price_table()
    relprice_max = 2
    price_table = transform(
        price_table,
        :trim_relprice => ByRow(min$relprice_max) => :trim_relprice_clamped,
    )
    price_table = subset(
        price_table,
        :make => ByRow(==(snakemake.wildcards["manufacturer"])),
        view = true,
    )

    histogram(
        price_table.trim_rank,
        weights = price_table.n,
        label = nothing,
        bins = 0:0.2:1.2,
        size = (800, 600),
    )
    xlabel!("Trim Rank")
    ylabel!("# Registrations")
    xlims!(0, 1.2)

    png(snakemake.output["rank"])

    histogram(
        price_table.trim_relprice_clamped,
        weights = price_table.n,
        label = nothing,
        bins = 1:0.05:(1.05relprice_max),
        size = (800, 600),
    )
    xlabel!("Trim Relative Price")
    ylabel!("# Registrations")
    xlims!(1, 1.05relprice_max)

    png(snakemake.output["relprice"])
end
