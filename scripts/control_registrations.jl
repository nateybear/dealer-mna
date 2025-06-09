using DealerMnA
using DrWatson
import JLD2
import Arrow
using UUIDs

!isempty(ARGS) && snakemake!(ARGS[1])

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

    before_registrations = Empirics.control_registrations_before(t; dealer_uid)
    after_registrations = Empirics.control_registrations_after(t; dealer_uid)

    constant_dealers = before_registrations.dealer_uid ∩ after_registrations.dealer_uid

    subset!(before_registrations, :dealer_uid => ByRow(in(constant_dealers)))
    subset!(after_registrations, :dealer_uid => ByRow(in(constant_dealers)))

    Arrow.write(
        snakemake.output["before"],
        before_registrations,
    )
    Arrow.write(
        snakemake.output["after"],
        after_registrations,
    )
end
