using DealerMnA

using DataFrames
using Plots
using Lazy: @>

function (@main)(_)
    with_logging() do
        Dealers.load_from_disk!(snakemake.input["dealers_init"])
        counts = @> Dealers.State.df groupby(:gid) combine(nrow)
        histogram(
            counts.nrow,
            label=nothing,
            size=(600, 400),
        )
        png(snakemake.output[1])
    end

    return nothing
end
