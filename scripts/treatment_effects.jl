using DealerMnA
import JLD2
import Arrow
using DataFrames
import Lazy: @>
using UUIDs
using StatsBase: median
using FixedEffectModels: Term, ConstantTerm

!isempty(ARGS) && snakemake!(ARGS[1])

with_logging() do
    Dealers.load_from_disk!(snakemake.input["dealers_init"])
    registrations = DataFrame(Arrow.Table(snakemake.input["registrations"]))
    transaction_uids = read_csv(snakemake.input["transaction_uids"])
    mapcols!(ByRow(UUID), transaction_uids)
    dealer_demographics = DataFrame(Arrow.Table(snakemake.input["dealer_demographics"]))

    extras, coeftable =
        if !(snakemake.wildcards isa Dict) || !haskey(snakemake.wildcards, "cut")
            Empirics.event_study(registrations, dealer_demographics)
        elseif snakemake.wildcards["cut"] == "urban"
            dealer_demographics.urban =
                dealer_demographics.share_urban .> median(dealer_demographics.share_urban)
            Empirics.event_study(
                registrations,
                dealer_demographics,
                Term(:urban) + ConstantTerm(1),
                dealer_demographics,
                :dealer_uid => :uid,
            )
        elseif snakemake.wildcards["cut"] == "exiting_seller"
            transaction_uids.exiting_seller = map(transaction_uids.transaction_uid) do uid
                countlines("data/affiliated_dealerships/seller_group_$(uid).txt") == 0
            end
            Empirics.event_study(
                registrations,
                dealer_demographics,
                Term(:exiting_seller) + ConstantTerm(1),
                transaction_uids,
            )
        else
            error("Unknown cut: $(snakemake.wildcards["cut"])")
        end

    Arrow.write(
        snakemake.output["treatment_effects"],
        coeftable,
    )

    JLD2.save_object(snakemake.output["extra_stats"], extras)
end
