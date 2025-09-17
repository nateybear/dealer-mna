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

    # For closest dealers, dealer_uid wildcard is the closest dealer, 
    # and transacted_dealer_uid wildcard is the treated dealer from transaction_uids
    dealer_uid = UUID(snakemake.wildcards["dealer_uid"])

    # For closest dealers, we need a 3-part reg_uid: transaction_uid_treated_dealer_uid_closest_dealer_uid
    reg_uid = if haskey(snakemake.wildcards, "transacted_dealer_uid")
        transacted_dealer_uid = snakemake.wildcards["transacted_dealer_uid"]
        "$(tuid)_$(dealer_uid)_$(transacted_dealer_uid)"
    else
        "$(tuid)_$(dealer_uid)"
    end

    @info "Writing treated registrations for transaction $(t.uid)"
    before_registrations = Empirics.last_registrations_before(t; dealer_uid)
    after_registrations = Empirics.first_registrations_after(t; dealer_uid)

    before_registrations[!, :reg_uid] .= reg_uid
    after_registrations[!, :reg_uid] .= reg_uid

    Arrow.write(
        snakemake.output["before"],
        before_registrations,
    )
    Arrow.write(
        snakemake.output["after"],
        after_registrations,
    )
end
