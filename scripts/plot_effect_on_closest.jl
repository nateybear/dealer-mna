using DealerMnA
using Plots
using Distributions: pdf, Normal
using Optim: optimize
using StatsBase: mean, weights

import .Geometry: distance

isempty(ARGS) || snakemake!(ARGS[1])

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

    sales_extras = pluck(extras, :stat => ByRow(==(:total_sales)))
    fes = sales_extras.fes
    select!(fes, :reg_uid, :diff_in_diff => :sales_change)

    return innerjoin(transaction_uids, fes, on = :treated_reg_uid => :reg_uid)
end

test_derivative(diffs, V) = function (distance_log) # partial effect is β_sales + distance * β_{sales × distance}
    ∇ = [0.0, 0.0, 1.0, distance_log]
    derivative = sum(∇ .* diffs.diff)
    se_derivative = sqrt(∇'V * ∇)
    sign = if derivative > 0
        derivative > 1.96se_derivative ? :pos : :zero
    else
        derivative < -1.96se_derivative ? :neg : :zero
    end
    return (; derivative, se_derivative, sign)
end

"""
    spans_from_labels(xs::Vector, labels::Vector)

Given a grid `xs` and categorical `labels` (e.g. :pos, :neg, :zero),
return a vector of `(a, b, label)` tuples representing consecutive
intervals where the label is constant.
"""
function spans_from_labels(xs::Vector, labels::Vector)
    @assert length(xs) == length(labels)

    spans = []
    start = xs[1]
    current_label = labels[1]

    for i in 2:length(xs)
        if labels[i] != current_label
            # close span at previous point
            push!(spans, (start, xs[i - 1], current_label))
            # start new span
            start = xs[i]
            current_label = labels[i]
        end
    end

    # close last span
    push!(spans, (start, xs[end], current_label))

    return spans
end

function E_log_distance(df)
    E(h, subdf) = function (sales_change)
        u = (sales_change .- subdf.sales_change) ./ h
        w = pdf(Normal(), u)
        return mean(subdf.distance_log, weights(w))
    end
    return E(1e6, df) # optimal h calculation is unstable here
end

function fitted(diffs, V) # assume order is 1, distance_log, sales_change, interaction (which should always be true)
    diffs.interaction ==
    ["1", "distance_log", "sales_change", "distance_log & sales_change"] ||
        error("wrong variable order!")
    function (sales_change, distance_log)
        ∇ = [1.0, distance_log, sales_change, distance_log * sales_change]
        return (
            estimate = sum(∇ .* diffs.diff),
            se_estimate = sqrt(∇'V * ∇),
        )
    end
end

with_logging() do
    Dealers.load_from_disk!(snakemake.input["dealers_init"])
    coeftable = read_arrow(snakemake.input["treatment_effects"])
    transaction_uids = prepare_transaction_uids(
        read_csv(snakemake.input["transaction_uids"]),
        JLD2.load_object(snakemake.input["overall_extras"]),
    )
    extras = JLD2.load_object(snakemake.input["extras"])
    sales_extras = pluck(extras, :stat => ByRow(==(:total_sales)))

    diffs = @> coeftable begin
        subset(:T => ByRow(in([-3, 2])))
        sort(:T, rev = true)
        groupby(:interaction)
        combine(:total_sales => splat(-) => :diff)
    end

    V = let c = reshape(unique(coeftable.interaction), 1, :)
        coef_mask = coeftable.interaction .== c
        ∇ = (-(coeftable.T .== -3) .+ (coeftable.T .== 2)) .* coef_mask
        ∇'sales_extras.V * ∇
    end

    x_grid = DataFrame(
        sales_change = range(extrema(transaction_uids.sales_change)..., length = 500)
    )
    transform!(
        x_grid,
        :sales_change => ByRow(E_log_distance(transaction_uids)) => :distance_log,
    )
    transform!(
        x_grid,
        [:sales_change, :distance_log] => ByRow(fitted(diffs, V)) => AsTable,
        :distance_log => ByRow(test_derivative(diffs, V)) => AsTable,
    )

    scatter_df = innerjoin(transaction_uids, sales_extras.fes, on = :reg_uid)
    subset!(scatter_df, :diff_in_diff => ByRow(<(5e6) ∘ abs)) # knock a few outliers so lines look more sensible
    scatter(
        scatter_df.sales_change,
        scatter_df.diff_in_diff,
        label = nothing,
        size = (800, 600),
        xlabel = "Change in Transacted Sales (\$)",
        ylabel = "Change in Nearest Dealer Sales (\$)",
        left_margin = 5Plots.mm,
    )

    shaded_regions = spans_from_labels(x_grid.sales_change, x_grid.sign)
    for (xstart, xstop, sign) in shaded_regions
        @show xstart xstop sign
        if sign == :pos
            @info "POSSS"
            vspan!([xstart, xstop], alpha = 0.1, color = :green, label = nothing)
        elseif sign == :neg
            @info "NEGGG"
            vspan!([xstart, xstop], alpha = 0.1, color = :red, label = nothing)
        end
    end
    hline!([0.0], color = "black", label = nothing)
    vline!([0.0], color = "black", label = nothing)

    plot!(
        x_grid.sales_change,
        x_grid.estimate,
        ribbon = 1.96x_grid.se_estimate,
        color = :grey,
        label = nothing,
        linestrokewidth = 4,
        alpha = 0.4,
    )

    png(snakemake.output[1])
end
