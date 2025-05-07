using DealerMnA

import Arrow
using DataFrames
using Plots
using Lazy: @>
import StatsBase: std

function (@main)(_)
    with_logging() do
        event_study_df = DataFrame(Arrow.Table(snakemake.input[1]))
        statistic = Symbol(snakemake.wildcards["statistic"])

        sample_estimate = @> event_study_df subset(:B => ByRow(ismissing)) select(:T, statistic)

        bootstrap_estimates = subset(event_study_df, :B => ByRow(!ismissing))
        bootstrap_ses = @> bootstrap_estimates begin
            groupby(:T)
            combine(statistic => std ∘ skipmissing => :se)
        end

        plot_df = dropmissing(innerjoin(sample_estimate, bootstrap_ses, on=:T))

        plot(
            plot_df.T,
            plot_df[!, statistic],
            yerr=1.96plot_df.se,
            msc=1,
            label=nothing,
            color=1
        )

        xticks!(-6:5)
        scatter!(plot_df.T, plot_df[!, statistic], label=nothing, color=1)
        hline!([0], color=:black, label=nothing)
        vline!([-0.5], color=:black, label=nothing)

        png(snakemake.output[1])
    end
    return nothing
end
