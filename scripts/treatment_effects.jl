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
    registrations = read_arrow([
        snakemake.input["treated_registrations"],
        snakemake.input["control_registrations"],
    ])
    dealer_demographics = read_arrow(snakemake.input["dealer_demographics"])

    Config = Empirics.parse_event_study_config(
        snakemake.wildcards["specification"]
    )

    if !haskey(snakemake.wildcards, "cut")
        # overall treatment effect, which we split by concentration change
        transaction_uids = read_csv(snakemake.input["transaction_uids"])
        mapcols!(ByRow(UUID), transaction_uids)
        transactions = JLD2.load_object(
            snakemake.input["transactions_for_analysis"]
        )
        transaction_uids = innerjoin(
            transaction_uids,
            DataFrame(
                transaction_uid = [t.uid for t in transactions],
                t = transactions,
            ),
            on = :transaction_uid,
        )
        transaction_uids.hhi_change = map(eachrow(transaction_uids)) do row
            Empirics.hhi(
                row.dealer_uid,
                [Dealers.eventtime(row.t, -3)],
                Dict(row.dealer_uid => row.t.buyer_gid),
            )
        end
        decreasers = subset(transaction_uids, :hhi_change => ByRow(<=(0)))
        increasers = subset(transaction_uids, :hhi_change => ByRow(>=(0)))

        increasing_registrations = subset(
            registrations, :dealer_uid => ByRow(!in(decreasers.dealer_uid))
        )
        increasing_config = Config(
            registrations = increasing_registrations,
            dealer_demographics = dealer_demographics,
        )
        JLD2.save_object(
            snakemake.output["increasing"],
            Empirics.event_study(increasing_config),
        )

        decreasing_registrations = subset(
            registrations, :dealer_uid => ByRow(!in(increasers.dealer_uid))
        )
        decreasing_config = Config(
            registrations = decreasing_registrations,
            dealer_demographics = dealer_demographics,
        )
        JLD2.save_object(
            snakemake.output["decreasing"],
            Empirics.event_study(decreasing_config),
        )

        overall_config = Config(
            registrations = registrations,
            dealer_demographics = dealer_demographics,
        )
        JLD2.save_object(
            snakemake.output["overall"],
            Empirics.event_study(overall_config),
        )
    elseif snakemake.wildcards["cut"] == "urban"
        dealer_demographics.urban =
            dealer_demographics.share_urban .> median(dealer_demographics.share_urban)
        config = Config(
            registrations = registrations,
            dealer_demographics = dealer_demographics,
            interaction_style = Empirics.WithIncludedInteraction(Term(:urban) + ContantTerm(1))
        )
        JLD2.save_object(snakemake.output[1], Empirics.event_study(config))
    elseif snakemake.wildcards["cut"] == "exiting_seller"
        transaction_uids = read_csv(snakemake.input["transaction_uids"])
        mapcols!(ByRow(UUID), transaction_uids)
        transaction_uids.exiting_seller = map(transaction_uids.transaction_uid) do uid
            countlines("data/affiliated_dealerships/seller_group_$(uid).txt") == 0
        end
        config = Config(
            registrations = registrations,
            dealer_demographics = dealer_demographics,
            interaction_style = Empirics.WithAdditionalInteraction(
                term = Term(:exiting_seller) + ConstantTerm(1),
                additional_data = transaction_uids,
            ),
        )
        JLD2.save_object(snakemake.output[1], Empirics.event_study(config))
    else
        error("Unknown cut: $(snakemake.wildcards["cut"])")
    end
end
