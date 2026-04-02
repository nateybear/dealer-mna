using DealerMnA
using Arrow

snakemake!()

with_logging() do
    Arrow.write(
        snakemake.output[1],
        Registrations.make_price_table(),
    )
end
