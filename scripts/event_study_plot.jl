using DealerMnA

import Arrow
using DataFrames
using Plots
using UUIDs
using Lazy: @>
import StatsBase: std, mean, sample, median

resample(x) =
    let x = collect(x)
        sample(x, length(x))
    end

function event_study_plot!(plot_df, statistic)
    (first_group, rest...) = groupby(plot_df, "group" in names(plot_df) ? :group : [])

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

function (@main)(_)
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

            if cut == "grand_mean"
                sample_estimate = @> event_study_df begin
                    subset(:B => ByRow(ismissing))
                    select(:T, statistic)
                    groupby(:T)
                    combine(statistic => mean ∘ skipmissing => statistic)
                end

                bootstrap_estimates = @> event_study_df begin
                    subset(:B => ByRow(!ismissing))
                    groupby([:B, :T])
                    combine(statistic => mean ∘ skipmissing => statistic)
                end

                bootstrap_ses = @> bootstrap_estimates begin
                    groupby(:T)
                    combine(statistic => std ∘ skipmissing => :se)
                end

                plot_df = dropmissing(innerjoin(sample_estimate, bootstrap_ses, on = :T))

                event_study_plot!(plot_df, statistic)

            elseif cut in names(dealer_demographics)
                mid = median(dealer_demographics[!, cut])
                cut_name = titlecase(replace(cut, r"_" => " "))
                event_study_df = @> event_study_df begin
                    innerjoin(dealer_demographics, on = :dealer_uid => :uid)
                    transform(cut => ByRow(>(mid)) => :group)
                    transform(
                        :group =>
                            ByRow() do x
                                x ? "Higher $cut_name" : "Lower $cut_name"
                            end =>
                                :group,
                    )
                end
                sample_estimate = @> event_study_df begin
                    subset(:B => ByRow(ismissing))
                    select(:T, :group, statistic)
                    groupby([:T, :group])
                    combine(statistic => mean ∘ skipmissing => statistic)
                end

                bootstrap_estimates = @> event_study_df begin
                    subset(:B => ByRow(!ismissing))
                    groupby([:B, :T, :group])
                    combine(statistic => mean ∘ skipmissing => statistic)
                end

                bootstrap_ses = @> bootstrap_estimates begin
                    groupby([:T, :group])
                    combine(statistic => std ∘ skipmissing => :se)
                end

                plot_df = dropmissing(
                    innerjoin(sample_estimate, bootstrap_ses, on = [:T, :group])
                )

                event_study_plot!(plot_df, statistic)
            end

        else # this is just plotting one dealership
            event_study_df = DataFrame(Arrow.Table(snakemake.input[1]))

            sample_estimate = @> event_study_df begin
                subset(:B => ByRow(ismissing))
                select(:T, statistic)
            end

            bootstrap_estimates = subset(event_study_df, :B => ByRow(!ismissing))
            bootstrap_ses = @> bootstrap_estimates begin
                groupby(:T)
                combine(statistic => std ∘ skipmissing => :se)
            end

            plot_df = dropmissing(innerjoin(sample_estimate, bootstrap_ses, on = :T))

            event_study_plot!(plot_df, statistic)
        end
    end
    return nothing
end
