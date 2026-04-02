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

function prepare_transaction_uids(transaction_uids, extras, group)
    transform!(extras, :dealer_uid => ByRow(UUID), renamecols = false) # might have been c
    transform!(
        transaction_uids,
        [:transaction_uid, :dealer_uid] =>
            ByRow(
                (tuid, duid) ->
                    readlines("data/closest_dealerships/$(tuid)_$(duid)_$(group).txt") |>
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
    subset!(transaction_uids, :same_group => ByRow(!))
    transform!(
        transaction_uids,
        :distance => ByRow(x -> iszero(x) ? log(1e-6) : log(x)) => :distance_log,
    ) # some dealers with distance zero (they are next to each other but not co-owned)

    select!(extras, :dealer_uid, :diff_in_diff => :sales_change)

    return @> transaction_uids begin
        innerjoin(extras, on = :dealer_uid)
        groupby(:closest_dealer_uid)
        subset(:distance => d -> d .== minimum(d)) # take closest if dealer appears twice
    end
end

snakemake!()

with_logging() do
    Dealers.load_from_disk!(snakemake.input["dealers_init"])
    transactions = JLD2.load_object(snakemake.input["transactions_for_analysis"])
    transactions = DataFrame(
        t = transactions,
        uid = [t.uid for t in transactions],
    )
    transaction_uids = prepare_transaction_uids(
        read_csv(snakemake.input["transaction_uids"]),
        JLD2.load_object(snakemake.input["treatment_effect_overall"])[:total_sales].extras,
        snakemake.wildcards["group"],
    )
    registrations = read_arrow([
        snakemake.input["treated_registrations"],
        snakemake.input["control_registrations"],
    ])

    # eliminate doubled registrations if a dealer is closest twice
    retained_τ_table = mapreduce(vcat, eachrow(transaction_uids)) do r
        transaction = pluck(transactions, :uid => ByRow(==(r.transaction_uid))).t
        return DataFrame(
            dealer_uid = r.closest_dealer_uid,
            τ = collect(-4:3),
            t = Dealers.eventtime.(Ref(transaction), -4:3),
        )
    end
    registrations = vcat(
        semijoin(# treated registrations
            subset(registrations, :D),
            retained_τ_table,
            on = [:dealer_uid, :t, :τ],
        ),
        antijoin(# control registrations
            subset(registrations, :D => ByRow(!)),
            retained_τ_table,
            on = :dealer_uid,
        ),
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
        Empirics.event_study(config, :total_sales => Empirics.STATISTICS[:total_sales]),
    )
end
