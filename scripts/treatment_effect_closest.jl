using DealerMnA
using FixedEffectModels: Term, ConstantTerm

import .Geometry: distance

function miles(d1::UUID, d2::UUID)
    dealers = subset(Dealers.State.df, :uid => ByRow(in((d1, d2))), view = true)
    return Geometry.miles(dealers.geometry...)
end

function same_group(tuid, dealer_uid)
    t = first(t for t in Dealers.Events if t.uid == tuid)
    Dealers.timetravel(t) do
        dealers = subset(Dealers.State.df, :uid => ByRow(==(dealer_uid)), view = true)
        return !isempty(dealers.gid ∩ (t.buyer_gid, t.seller_gid))
    end
end

function prepare_transaction_uids(transaction_uids, extras)
    transform!(
        transaction_uids,
        [:transaction_uid, :dealer_uid] =>
            ByRow(
                (tuid, duid) ->
                    readlines("data/closest_dealerships/$(tuid)_$(duid).txt") |>
                    enumerate |>
                    collect,
            ) => :closest_dealer_uid,
    )
    transaction_uids = DataFrames.flatten(transaction_uids, :closest_dealer_uid)
    transaction_uids.closest_dealer_index = first.(transaction_uids.closest_dealer_uid)
    transaction_uids.closest_dealer_uid = last.(transaction_uids.closest_dealer_uid)

    transform!(
        transaction_uids,
        [:dealer_uid, :transaction_uid, :closest_dealer_uid] .=> ByRow(UUID),
        renamecols = false,
    )
    transform!(
        transaction_uids,
        [:dealer_uid, :closest_dealer_uid] => ByRow(miles) => :distance,
        [:transaction_uid, :closest_dealer_uid] =>
            ByRow(same_group) => :same_group,
    )
    transform!(transaction_uids, :distance => ByRow(log))
    subset!(transaction_uids, :same_group => ByRow(!))

    select!(extras, :dealer_uid => :treated_dealer_uid, :diff_in_diff => :sales_change)

    return innerjoin(transaction_uids, extras, on = :dealer_uid)
end

!isempty(ARGS) && snakemake!(ARGS[1])

with_logging() do
    Dealers.load_from_disk!(snakemake.input["dealers_init"])
    transaction_uids = prepare_transaction_uids(
        read_csv(snakemake.input["transaction_uids"]),
        JLD2.load_object(snakemake.input["treatment_effect_overall"])[:total_sales].extras,
    )
    registrations = read_arrow(
        snakemake.input["treated_registrations"], snakemake.input["control_registrations"]
    )
    dealer_demographics = read_arrow(snakemake.input["dealer_demographics"])
    heterogeneity_term = ConstantTerm(1) + Term(:distance_log) * Term(:sales_change)

    config = let Config = Empirics.parse_event_study_config(snakemake.wildcards["specification"])
        Config(
            registrations = registrations,
            dealer_demographics = dealer_demographics,
            interaction_style = Empirics.WithAdditionalInteraction(
                term = heterogeneity_term,
                additional_data = select(
                    transaction_uids,
                    Not(:dealer_uid),
                    :closest_dealer_uid => :dealer_uid,
                    copycols = false,
                ),
            ),
        )
    end

    JLD2.save_object(
        snakemake.output[1],
        Empirics.event_study(config, :sales_change => Empirics.DEFAULT_STATISTICS[:sales_change]),
    )
end
