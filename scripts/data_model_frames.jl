using DealerMnA

snakemake!()

with_logging() do
    Dealers.load_from_disk!(snakemake.input["dealers_init"])
    JLD2.save_object(snakemake.output[1], Demand.load_model_frames())
    return nothing
end
