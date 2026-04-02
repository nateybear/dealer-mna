using DealerMnA

using DataFrames
using Plots
using Lazy: @>

snakemake!()

with_logging() do
    Dealers.load_from_disk!(snakemake.input["dealers_init"])
    counts = @> Dealers.State.df groupby(:gid) combine(nrow)
    histogram(
        counts.nrow,
        label = nothing,
        size = (600, 400),
    )
    png(snakemake.output["histogram"])

    sort!(counts, :nrow)
    counts.cumulative = cumsum(counts.nrow)
    counts = @> counts groupby(:nrow) combine(:cumulative => last, renamecols=false)
    plot(
        counts.nrow,
        counts.cumulative,
        label = nothing,
        size = (800, 600),
        linetype=:steppost,
        xlabel="Dealer Group Size",
        ylabel="Cumulative Number of Dealers",
        left_margin = 5Plots.mm,
    )
    png(snakemake.output["cumulative"])
end
