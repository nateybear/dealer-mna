using DealerMnA

import Arrow
using DataFrames
using Plots
using Lazy: @>
import StatsBase: std, mean

function event_study_plot!(plot_df, statistic)
    plot(
        plot_df.T,
        plot_df[!, statistic],
        yerr=1.96plot_df.se,
        msc=1,
        label=nothing,
        color=1
    )

    xticks!(Empirics.VALID_PERIODS)
    scatter!(plot_df.T, plot_df[!, statistic], label=nothing, color=1)
    hline!([0], color=:black, label=nothing)
    vline!([-0.5], color=:black, label=nothing)

    png(snakemake.output[1])
end

function (@main)(_)
    with_logging() do
        statistic = Symbol(snakemake.wildcards["statistic"])

        if haskey(snakemake.wildcards, "cut") # this is plotting an aggregate
            @info "This is snakemake" snakemake.input
            event_study_df = mapreduce(DataFrame ∘ Arrow.Table, vcat, snakemake.input["treatment_effects"])

            # for now, just grand mean
            sample_estimate = @> event_study_df subset(:B => ByRow(ismissing)) select(:T, statistic) groupby(:T) combine(statistic => mean ∘ skipmissing => statistic)

            bootstrap_estimates = @> event_study_df subset(:B => ByRow(!ismissing)) groupby([:B, :T]) combine(statistic => mean ∘ skipmissing => statistic)
            bootstrap_ses = @> bootstrap_estimates begin
                groupby(:T)
                combine(statistic => std ∘ skipmissing => :se)
            end

            plot_df = dropmissing(innerjoin(sample_estimate, bootstrap_ses, on=:T))

            event_study_plot!(plot_df, statistic)

        else # this is just plotting one dealership
            event_study_df = DataFrame(Arrow.Table(snakemake.input[1]))

            sample_estimate = @> event_study_df subset(:B => ByRow(ismissing)) select(:T, statistic)

            bootstrap_estimates = subset(event_study_df, :B => ByRow(!ismissing))
            bootstrap_ses = @> bootstrap_estimates begin
                groupby(:T)
                combine(statistic => std ∘ skipmissing => :se)
            end

            plot_df = dropmissing(innerjoin(sample_estimate, bootstrap_ses, on=:T))

            event_study_plot!(plot_df, statistic)
        end
    end
    return nothing
end
