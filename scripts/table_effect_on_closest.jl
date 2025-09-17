using DealerMnA
using FixedEffectModels
using RegressionTables
using Unitful
using Plots
using Humanize: digitsep
using Intervals

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
        transaction_uids.dealer_uid .* "_" .*
        transaction_uids.closest_dealer_uid

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
        [:reg_uid, :dealer_uid],
    )

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
    regtable(
        models...,
        file = snakemake.output["table_regression"],
        drop = ["(Intercept)"],
        render = LatexTable(),
        print_fe_section = false,
    )
    remove_control_line!()

    # draw scatter and best fit for sales
    (b, m, _) = coef(models[1])
    plot_df = subset(
        transaction_uids, :treated_total_sales => ByRow(in(-2_500_000 .. 5_000_000))
    )
    p = scatter(
        plot_df.treated_total_sales,
        plot_df.closest_total_sales,
        group = plot_df.closest_dealer_index,
        palette = :Set2,
        xlabel = "Δ Quarterly Sales, Treated",
        ylabel = "Δ Quarterly Sales, Closest",
        label = ["1st" "2nd" "3rd"] .* "-closest",
        margin = 2Plots.mm,
        legend = :topright,
        size = (700, 525),
    )

    (breaks, _) = xticks(p)[1]
    labels = map(breaks) do b
        readable = digitsep(abs(round(Int64, b)))
        b < 0 ? "(\$$readable)" : "\$$(readable)"
    end
    xticks!(p, (breaks, labels))

    (breaks, _) = yticks(p)[1]
    labels = map(breaks) do b
        readable = digitsep(abs(round(Int64, b)))
        b < 0 ? "(\$$readable)" : "\$$(readable)"
    end
    yticks!(p, (breaks, labels))

    Plots.abline!(p, m, b, color = :black, linewidth = 2, label = "Best fit line")
    hline!(p, [0.0], color = :black, label = nothing, linestyle = :dot)

    png(snakemake.output["scatter_sales"])
end
