using DealerMnA

snakemake!()

with_logging() do
    Dealers.load_from_disk!(snakemake.input["dealers_init"])
    Dealers.State.df = semijoin(
        Dealers.State.df,
        Registrations.registrations(),
        on = :uid => :dealer_uid,
    )
    Dealers.compile_diffs!()
    Dealers.write_to_disk!(snakemake.output[1])
    return nothing
end
