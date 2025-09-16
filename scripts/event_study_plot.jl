using DealerMnA

import Arrow
using DataFrames
using Plots
using UUIDs
using Lazy: @>
import StatsBase: std, mean, sample, median

!isempty(ARGS) && snakemake!(ARGS[1])

function add_zero(df, statistic)
    below_zero = subset(df, :T => ByRow(<(0)))
    above_zero = subset(df, :T => ByRow(>(0)))
    return vcat(
        below_zero,
        DataFrame([:T, :se, statistic] .=> 0.0),
        above_zero,
        cols = :union,
    )
end

function event_study_plot!(plot_df, statistic, adjustment = 0)
    plot() # create an empty plot
    hline!([0], color = :black, label = nothing)
    vline!([-0.5], color = :black, label = nothing)

    (first_group, rest...) = groupby(plot_df, "group" in names(plot_df) ? :group : [])

    first_label = isempty(rest) ? nothing : only(unique(first_group.group))
    first_group = add_zero(first_group, statistic)
    first_group[!, statistic] .+= adjustment

    plot!(
        first_group.T,
        first_group[!, statistic],
        yerr = 1.96first_group.se,
        msc = 1,
        label = first_label,
        color = 1,
    )
    scatter!(first_group.T, first_group[!, statistic], label = nothing, color = 1)

    for (i, next_group) in enumerate(rest)
        label = only(unique(next_group.group))
        next_group = add_zero(next_group, statistic)
        next_group[!, statistic] .+= adjustment
        plot!(
            next_group.T,
            next_group[!, statistic],
            yerr = 1.96next_group.se,
            msc = i + 1,
            label = label,
            color = i + 1,
        )
        scatter!(next_group.T, next_group[!, statistic], label = nothing, color = i + 1)
    end

    ylims!(
        1.05min(adjustment, minimum(plot_df[!, statistic] .+ adjustment .- 1.96plot_df.se)),
        1.05max(adjustment, maximum(plot_df[!, statistic] .+ adjustment .+ 1.96plot_df.se)),
    )
    xticks!(Empirics.VALID_PERIODS)

    return png(snakemake.output[1])
end

with_logging() do
    statistic = Symbol(snakemake.wildcards["statistic"])
    df = DataFrame(Arrow.Table(snakemake.input["treatment_effect"]))
    rename!(df, Symbol("$(statistic)_se") => :se)
    event_study_plot!(df, statistic)
end
