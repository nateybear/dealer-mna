using DealerMnA
using Arrow

!isempty(ARGS) && snakemake!(ARGS[1])

with_logging() do
    Arrow.write(
        snakemake.output[1],
        Registrations.price_table(),
    )
end
