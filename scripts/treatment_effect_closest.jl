using DealerMnA
using FixedEffectModels: Term, ConstantTerm
using Unitful

import .Geometry: distance

function miles(d1::UUID, d2::UUID)
    dealers = subset(Dealers.State.df, :uid => ByRow(in((d1, d2))), view = true)
    return ustrip(uconvert(u"mi", distance(dealers.geometry...)))
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
    transaction_uids.reg_uid =
        transaction_uids.transaction_uid .* "_" .*
        transaction_uids.closest_dealer_uid .* "_" .*
        transaction_uids.dealer_uid

    transaction_uids.treated_reg_uid =
        transaction_uids.transaction_uid .* "_" .* transaction_uids.dealer_uid
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

    sales_extras = only(eachrow(subset(extras, :stat => ByRow(==(:total_sales)))))
    fes = sales_extras.fes
    select!(fes, :reg_uid, :diff_in_diff => :sales_change)

    return innerjoin(transaction_uids, fes, on = :treated_reg_uid => :reg_uid)
end

function all_registrations(transaction_uids)
    all_arrow_files = map(eachrow(transaction_uids)) do r
        [
            "data/treated/closest_dealers/last_registrations_before_$(r.reg_uid).arrow",
            "data/treated/closest_dealers/first_registrations_after_$(r.reg_uid).arrow",
            "data/control/closest_dealers/last_registrations_before_$(r.reg_uid).arrow",
            "data/control/closest_dealers/first_registrations_after_$(r.reg_uid).arrow",
        ]
    end
    return DataFrame(Arrow.Table(collect(Iterators.flatten(all_arrow_files))))
end

!isempty(ARGS) && snakemake!(ARGS[1])

with_logging() do
    Dealers.load_from_disk!(snakemake.input["dealers_init"])
    transaction_uids = prepare_transaction_uids(
        read_csv(snakemake.input["transaction_uids"]),
        JLD2.load_object(snakemake.input["extras"]),
    )
    registrations = all_registrations(transaction_uids)
    dealer_demographics = DataFrame(Arrow.Table(snakemake.input["dealer_demographics"]))
    heterogeneity_term = ConstantTerm(1) + Term(:distance_log) * Term(:sales_change)

    extras, coeftable = Empirics.event_study(
        registrations,
        dealer_demographics,
        heterogeneity_term,
        select(
            transaction_uids,
            Not(:dealer_uid),
            :closest_dealer_uid => :dealer_uid,
            copycols = false,
        ),
        how_join = [:reg_uid, :dealer_uid],
        statistics = [:total_sales],
    )

    JLD2.save_object(snakemake.output["extra_stats"], extras)
    Arrow.write(snakemake.output["treatment_effects"], coeftable)
end
