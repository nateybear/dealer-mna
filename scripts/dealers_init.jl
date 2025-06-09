using DealerMnA

!isempty(ARGS) && snakemake!(ARGS[1])

with_logging() do
    Dealers.init!()
    Dealers.write_to_disk!(snakemake.output[1])
end
