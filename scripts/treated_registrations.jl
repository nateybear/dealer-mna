using DealerMnA
using DrWatson
import JLD2
import Arrow
using UUIDs

function (@main)(_)
    with_logging() do
        Dealers.load_from_disk!(snakemake.input["dealers_init"])
        transactions = JLD2.load_object(snakemake.input["transactions_for_analysis"])
        t = first(t for t in transactions if t.uid == UUID(snakemake.wildcards["transaction_uid"]))
        dealer_uid = haskey(snakemake.wildcards, "dealer_uid") ? UUID(snakemake.wildcards["dealer_uid"]) : nothing
        @info "Writing treated registrations for transaction $(t.uid)"
        Arrow.write(
            snakemake.output["before"],
            Empirics.last_registrations_before(t; dealer_uid),
        )
        Arrow.write(
            snakemake.output["after"],
            Empirics.first_registrations_after(t; dealer_uid),
        )
    end
    return nothing
end
