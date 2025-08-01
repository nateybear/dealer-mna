using DealerMnA
using FixedEffectModels
using RegressionTables
using Unitful

import .Geometry: distance

function miles(d1::UUID, d2::UUID)
    dealers = subset(Dealers.State.df, :uid => ByRow(in((d1, d2))), view = true)
    return ustrip(uconvert(u"mi", distance(dealers.geometry...)))
end

function same_group(tuid, d1, d2)
    t = first(t for t in Dealers.Events if t.uid == tuid)
    Dealers.timetravel(t) do
        dealers = subset(Dealers.State.df, :uid => ByRow(in((d1, d2))), view = true)
        return allequal(dealers.gid)
    end
end

!isempty(ARGS) && snakemake!(ARGS[1])

with_logging() do
    Dealers.load_from_disk!(snakemake.input["dealers_init"])
    transaction_uids = read_csv(snakemake.input["transaction_uids"])
    transform!(
        transaction_uids,
        [:transaction_uid, :dealer_uid] =>
            ByRow(
                (tuid, duid) -> readline("data/closest_dealerships/$(tuid)_$(duid).txt")
            ) => :closest_dealer_uid,
    )
    transform!(
        transaction_uids,
        names(transaction_uids) .=> ByRow(UUID),
        renamecols = false,
    )
    transform!(
        transaction_uids,
        [:dealer_uid, :closest_dealer_uid] => ByRow(miles) => :distance,
        [:transaction_uid, :dealer_uid, :closest_dealer_uid] =>
            ByRow(same_group) => :same_group,
    )
    subset!(transaction_uids, :same_group => ByRow(!))
    te_cols = [
        :total_sales, :mean_price, :total_units, :trim_rank
    ]
    for col in te_cols
        function te(group, transaction_uid, dealer_uid)
            return first(Empirics.treatment_effect(col, group; transaction_uid, dealer_uid))
        end
        transform!(
            transaction_uids,
            [:transaction_uid, :dealer_uid] =>
                ByRow(te$("treated_dealers")) => Symbol("treated_$col"),
            [:transaction_uid, :closest_dealer_uid] =>
                ByRow(te$("closest_dealers")) => Symbol("closest_$col"),
        )
    end

    models = map(te_cols) do col
        form =
            Term(Symbol("closest_$col")) ~
                Term(Symbol("treated_$col")) +
                Term(:distance) & Term(Symbol("treated_$col"))
        reg(transaction_uids, form)
    end
    regtable(models..., file = snakemake.output[1], drop = ["(Intercept)"])
end
