using DealerMnA
using DrWatson
import JLD2
import Arrow
using UUIDs

function (@main)(_)
    with_logging() do
        Dealers.load_from_disk!(snakemake.input["dealers_init"])
        transactions = JLD2.load_object(snakemake.input["transactions_for_analysis"])
        t = first(t for t in transactions if t.uid == UUID(snakemake.wildcards["uid"]))
        Arrow.write(
            snakemake.output["before"],
            Empirics.control_registrations_before(t),
        )
        Arrow.write(
            snakemake.output["after"],
            Empirics.control_registrations_after(t),
        )
    end
    return nothing
end
