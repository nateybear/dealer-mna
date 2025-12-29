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

function prepare_transaction_uids(transaction_uids, extras, group)
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
    transform!(
        transaction_uids,
        :distance => ByRow(x -> iszero(x) ? log(1e-6) : log(x)) => :distance_log,
    )
    subset!(transaction_uids, :same_group => ByRow(!))

    select!(extras, :dealer_uid, :diff_in_diff => :sales_change)

    return @> transaction_uids begin
        innerjoin(extras, on = :dealer_uid)
        groupby(:closest_dealer_uid)
        subset(:distance => d -> d .== minimum(d)) # take closest if dealer appears twice
    end
end

function test_derivative(results, distance_log) # partial effect is β_sales + distance * β_{sales × distance}
    ∇ = [0.0, 0.0, 1.0, distance_log]
    derivative = sum(∇ .* results.did_estimate.array)
    se_derivative = sqrt(∇'results.did_stderr.array * ∇)
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

function fitted(results, sales_change, distance_log)
    ∇ = [1.0, distance_log, sales_change, distance_log * sales_change]
    return (
        estimate = sum(∇ .* results.did_estimate.array),
        se_estimate = sqrt(∇'results.did_stderr.array * ∇),
    )
end

with_logging() do
    Dealers.load_from_disk!(snakemake.input["dealers_init"])

    results = JLD2.load_object(snakemake.input["treatment_effect"])

    transaction_uids = prepare_transaction_uids(
        read_csv(snakemake.input["transaction_uids"]),
        JLD2.load_object(snakemake.input["overall_effects"])[:total_sales].extras,
        snakemake.wildcards["group"],
    )

    x_grid = DataFrame(
        sales_change = range(1.05 .* extrema(transaction_uids.sales_change)..., length = 500)
    )
    transform!(
        x_grid,
        :sales_change => ByRow(E_log_distance(transaction_uids)) => :distance_log,
    )
    transform!(
        x_grid,
        [:sales_change, :distance_log] => ByRow(fitted$(results)) => AsTable,
        :distance_log => ByRow(test_derivative$(results)) => AsTable,
    )

    scatter_df = innerjoin(
        transaction_uids,
        results.extras,
        on = :closest_dealer_uid => :dealer_uid,
    )
    subset!(scatter_df, :diff_in_diff => ByRow(<(5e6) ∘ abs)) # knock a few outliers so lines look more sensible
    scatter(
        scatter_df.sales_change,
        scatter_df.diff_in_diff,
        label = nothing,
        size = (800, 600),
        xlabel = "Change in Transacted Sales (\$)",
        ylabel = "Change in Nearest ($(titlecase(snakemake.wildcards["group"]))) Dealer Sales (\$)",
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

    avg_deriv = test_derivative(
        results,
        mean(transaction_uids.distance_log),
    )

    # find best corner to place annotation
    x_pos = extrema(scatter_df.sales_change)
    y_pos = extrema(scatter_df.diff_in_diff)
    corners = collect(Iterators.product(x_pos, y_pos))
    text_positions = collect(Iterators.product((:left, :right), (:bottom, :top)))
    counts = map(corners) do (x_c, y_c)
        x_counts = x_c < mean(scatter_df.sales_change) ?
            scatter_df.sales_change .<= mean(scatter_df.sales_change) :
            scatter_df.sales_change .> mean(scatter_df.sales_change)
        y_counts = y_c < mean(scatter_df.diff_in_diff) ?
            scatter_df.diff_in_diff .<= mean(scatter_df.diff_in_diff) :
            scatter_df.diff_in_diff .> mean(scatter_df.diff_in_diff)
        return sum(x_counts .& y_counts)
    end
    best_corner = corners[argmin(counts)]
    best_text_pos = text_positions[argmin(counts)]

    annotate!(
        best_corner...,
        text(
            "Avg. Deriv.: $(round(avg_deriv.derivative, digits = 4)) ($(round(avg_deriv.se_derivative, digits = 4)))",
            12,
            :black,
            best_text_pos...,
        ),
    )

    png(snakemake.output[1])
end
