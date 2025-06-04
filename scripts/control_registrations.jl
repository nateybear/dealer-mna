using DealerMnA
using DrWatson
import JLD2
import Arrow
using UUIDs

function (@main)(_)
    with_logging() do
        Dealers.load_from_disk!(snakemake.input["dealers_init"])
        transactions = JLD2.load_object(snakemake.input["transactions_for_analysis"])
        tuid = UUID(snakemake.wildcards["transaction_uid"])
        t = first(t for t in transactions if t.uid == tuid)
        dealer_uid = if haskey(snakemake.wildcards, "dealer_uid")
            UUID(snakemake.wildcards["dealer_uid"])
        else
            nothing
        end
        Arrow.write(
            snakemake.output["before"],
            Empirics.control_registrations_before(t; dealer_uid),
        )
        Arrow.write(
            snakemake.output["after"],
            Empirics.control_registrations_after(t; dealer_uid),
        )
    end
    return nothing
end
