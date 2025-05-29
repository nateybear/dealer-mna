using DealerMnA
using Arrow

function (@main)(_)
    with_logging() do
        Arrow.write(
            snakemake.output[1],
            Registrations.price_table(),
        )
    end
    return nothing
end
