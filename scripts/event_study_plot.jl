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

function add_zero!(df, statistic)
    return append!(df, DataFrame([:T, :se, statistic] .=> 0.0), cols = :union)
end

function event_study_plot!(plot_df, statistic)
    (first_group, rest...) = groupby(plot_df, "group" in names(plot_df) ? :group : [])

    add_zero!(first_group)
    first_label = isempty(rest) ? nothing : only(unique(first_group.group))

    plot(
        first_group.T,
        first_group[!, statistic],
        yerr = 1.96first_group.se,
        msc = 1,
        label = first_label,
        color = 1,
    )
    scatter!(first_group.T, first_group[!, statistic], label = nothing, color = 1)

    for (i, next_group) in enumerate(rest)
        add_zero!(next_group)
        label = only(unique(next_group.group))
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
        1.05minimum(plot_df[!, statistic] .- 1.96plot_df.se),
        1.05maximum(plot_df[!, statistic] .+ 1.96plot_df.se),
    )
    xticks!(Empirics.VALID_PERIODS)
    hline!([0], color = :black, label = nothing)
    vline!([-0.5], color = :black, label = nothing)

    return png(snakemake.output[1])
end

with_logging() do
    statistic = Symbol(snakemake.wildcards["statistic"])

        if haskey(snakemake.wildcards, "cut") # this is plotting an aggregate
            cut = snakemake.wildcards["cut"]
            event_study_df = mapreduce(vcat, snakemake.input["treatment_effects"]) do fname
                df = DataFrame(Arrow.Table(fname))
                ### extract transaction_uid and dealer_uid from the pathname
                df.transaction_uid .= UUID(basename(dirname(fname)))
                df.dealer_uid .= UUID(replace(basename(fname), r"\.arrow$" => ""))
                df
            end

            dealer_demographics = DataFrame(
                Arrow.Table(snakemake.input["dealer_demographics"])
            )

            # combine as if independent random variables
            combine_std_error(s) = sqrt(mean(s .^ 2))

            if cut == "grand_mean"
                plot_df = @> event_study_df begin
                    groupby(:T)
                    combine(
                        statistic => mean => statistic,
                        "$(statistic)_se" => combine_std_error => :se,
                    )
                end

                event_study_plot!(plot_df, statistic)

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
                    )
                end

                event_study_plot!(plot_df, statistic)
            else
                error("Unknown cut $(cut)")
            end

        else # this is just plotting one dealership
            plot_df = @> snakemake.input[1] begin
                Arrow.Table
                DataFrame
                rename("$(statistic)_se" => :se)
            end

            event_study_plot!(plot_df, statistic)
        end
    end
end
