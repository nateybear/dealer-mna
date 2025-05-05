using DealerMnA
using Arrow
using DataFrames

function (@main)(_)
    with_logging() do
        Dealers.load_from_disk!(snakemake.input["dealers_init"])
        out = mapreduce(vcat, eachrow(Dealers.State.df)) do d
            DataFrame(
                uid=d.uid,
                total_population=Empirics.total_population(d),
                median_income=Empirics.median_income(d),
                share_urban=Empirics.share_urban(d),
                share_white=Empirics.share_white(d),
                share_black=Empirics.share_black(d),
                share_hispanic=Empirics.share_hispanic(d),
            )
        end
        Arrow.write(
            snakemake.output[1],
            out
        )
    end
    return nothing
end
