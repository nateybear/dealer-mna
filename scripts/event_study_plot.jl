using DealerMnA

import Arrow
using DataFrames
using Plots
using UUIDs
using Lazy: @>
import StatsBase: std, mean, sample, median

!isempty(ARGS) && snakemake!(ARGS[1])

resample(x) =
    let x = collect(x)
        sample(x, length(x))
    end

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

    if haskey(snakemake.wildcards, "cut") # this is plotting an aggregate
        cut = snakemake.wildcards["cut"]
        event_study_df = mapreduce(
            vcat, zip(snakemake.input["treatment_effects"], snakemake.input["extras"])
        ) do (tes, extras)
            df = DataFrame(Arrow.Table(tes))
            ### extract transaction_uid and dealer_uid from the pathname
            df.transaction_uid .= UUID(basename(dirname(fname)))
            df.dealer_uid .= UUID(replace(basename(fname), r"\.arrow$" => ""))
            extras = JLD2.load_object(extras)
            extras.delta = extras.fe_treated .- extras.fe_synthetic
            extras = @> extras select(:stat, :delta) unstack(:stat, :delta)
            rename!(col -> "$(col)_adjustment", extras)
            @> df begin
                subset(Symbol("$(statistic)_se") => ByRow(!isnan)) # that pesky underperforming dealer
                hcat(extras)
            end
        end

        dealer_demographics = DataFrame(
            Arrow.Table(snakemake.input["dealer_demographics"])
        )

        # combine as if independent random variables (var(mean) = var(sum) / n^2)
        combine_std_error(s) = sqrt(sum(s .^ 2) / length(s)^2)

        if cut == "grand_mean"
            plot_df = @> event_study_df begin
                groupby(:T)
                combine(
                    statistic => mean => statistic,
                    "$(statistic)_se" => combine_std_error => :se,
                    "$(statistic)_adjustment" => mean => :adjustment,
                )
            end

            event_study_plot!(plot_df, statistic, mean(plot_df.adjustment))

        elseif cut in names(dealer_demographics)
            mid = median(dealer_demographics[!, cut])
            cut_name = titlecase(replace(cut, r"_" => " "))
            add_group(x) = x > mid ? "Higher $cut_name" : "Lower $cut_name"

            event_study_df = @> event_study_df begin
                innerjoin(dealer_demographics, on = :dealer_uid => :uid)
                transform(cut => ByRow(add_group) => :group)
            end

            plot_df = @> event_study_df begin
                groupby([:T, :group])
                combine(
                    statistic => mean => statistic,
                    "$(statistic)_se" => combine_std_error => :se,
                    "$(statistic)_adjustment" => mean => :adjustment,
                )
            end

            event_study_plot!(plot_df, statistic, mean(plot_df.adjustment))
        else
            error("Unknown cut $(cut)")
        end

    else # this is just plotting one dealership
        plot_df = @> snakemake.input["treatment_effects"] begin
            Arrow.Table
            DataFrame
            rename("$(statistic)_se" => :se)
            subset(:se => ByRow(!isnan)) # that pesky underperforming dealer
        end

        adjustments = JLD2.load_object(snakemake.input["extra_stats"])
        adjustment = let r = first(r for r in eachrow(adjustments) if r.stat == statistic)
            r.fe_treated - r.fe_synthetic
        end

        event_study_plot!(plot_df, statistic, adjustment)
    end
end
