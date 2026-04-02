using DealerMnA

snakemake!()

with_logging() do
    Dealers.init!()
    Dealers.write_to_disk!(snakemake.output[1])
end
