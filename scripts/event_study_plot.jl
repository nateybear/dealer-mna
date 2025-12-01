using DealerMnA

using Plots
import StatsBase: std, mean, sample, median
using NamedArrays

!isempty(ARGS) && snakemake!(ARGS[1])

function add_zero(periods, estimates, stderrs)
    if 0 in es_result.periods
        return DataFrame(
            T = collect(periods),
            estimate = collect(estimates),
            se = collect(stderrs),
        )
    else
        sub_zero = findall(<(0), periods)
        sup_zero = findall(>(0), periods)
        return DataFrame(
            T = [periods[sub_zero]..., 0, periods[sup_zero]...],
            estimate = [estimates[sub_zero]..., 0.0, estimates[sup_zero]...],
            se = [stderrs[sub_zero]..., 0.0, stderrs[sup_zero]...],
        )
    end
end

function iter_series(es_result)
    if es_result.estimates[1] isa AbstractFloat
        return [(
            add_zero(
                es_result.periods,
                es_result.estimates,
                es_result.std_errs,
            ), "")]
    else # only do binary :)
        cut_name = titlecase(only(setdiff(names(es_result.estimates[1])[1], ["1"])))
        non_cut = add_zero(
            es_result.periods,
            getindex.(es_result.estimates, "1"),
            sqrt.(getindex.(es_result.std_errs, "1", "1")),
        )
        cut = add_zero(
            es_result.periods,
            sum.(es_result.estimates),
            sqrt.(sum.(es_result.std_errs)),
        )
        return [
            (non_cut, "Non $(cut_name)"),
            (cut, "$(cut_name)"),
        ]
    end
end

# only do a binary cut here :)
function event_study_plot!(es_result)
    plot() # create an empty plot
    hline!([0], color = :black, label = nothing)
    vline!([-0.5], color = :black, label = nothing)

    min_y = 0
    max_y = 0

    for (i, (plot_df, label)) in enumerate(iter_series(es_result))
        plot!(
            plot_df.T,
            plot_df.estimate,
            yerr = 1.96plot_df.se,
            msc = i,
            label = label,
            color = i,
        )
        scatter!(plot_df.T, plot_df.estimate, label = nothing, msc = i, color = i)
        min_y = min(min_y, minimum(plot_df.estimate .- 1.96plot_df.se))
        max_y = max(max_y, maximum(plot_df.estimate .+ 1.96plot_df.se))
    end

    ylims!(1.05min_y, 1.05max_y)
    xticks!(Empirics.VALID_PERIODS)

    return png(snakemake.output[1])
end

with_logging() do
    statistic = Symbol(snakemake.wildcards["statistic"])
    es_result::Empirics.EventStudyResult = JLD2.load_object(snakemake.input["treatment_effect"])[statistic]
    event_study_plot!(es_result)
end
