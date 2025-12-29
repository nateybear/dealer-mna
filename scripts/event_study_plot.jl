using DealerMnA

using Plots
import StatsBase: std, mean, sample, median
using NamedArrays
using Base.Iterators: flatmap

!isempty(ARGS) && snakemake!(ARGS[1])

function add_minus_one(periods, estimates, stderrs)
    if -1 in periods
        return DataFrame(
            T = collect(periods),
            estimate = collect(estimates),
            se = collect(stderrs),
        )
    else
        below = findall(<(-1), periods)
        above = findall(>(-1), periods)
        return DataFrame(
            T = [periods[below]..., -1, periods[above]...],
            estimate = [estimates[below]..., 0.0, estimates[above]...],
            se = [stderrs[below]..., 0.0, stderrs[above]...],
        )
    end
end

const Result{N, K} = Empirics.EventStudyResult{N, <:Union{Nothing, AbstractDataFrame}, <:K}

function iter_series(es_result::Result{N, AbstractFloat}, label = "") where {N}
    return [(
        add_minus_one(
            es_result.periods,
            es_result.estimates,
            es_result.std_errs,
        ), label)]
end

# only do binary :)
function iter_series(es_result::Result{N, NamedArray}, label = "") where {N}
    cut_name = titlecase(only(setdiff(names(es_result.estimates[1])[1], ["1"])))
    non_cut = add_minus_one(
        es_result.periods,
        getindex.(es_result.estimates, "1"),
        sqrt.(getindex.(es_result.std_errs, "1", "1")),
    )
    cut = add_minus_one(
        es_result.periods,
        sum.(es_result.estimates),
        sqrt.(sum.(es_result.std_errs)),
    )
    return [
        (non_cut, "$(label)Non $(cut_name)"),
        (cut, "$(label)$(cut_name)"),
    ]
end

function iter_series(es_results::AbstractVector{<:Empirics.EventStudyResult}, labels = nothing)
    if isnothing(labels)
        labels = fill("", length(es_results))
    end
    return flatmap(splat(iter_series), zip(es_results, labels))
end

# only do a binary cut here :)
function event_study_plot!(es_result, labels = "")
    plot() # create an empty plot
    hline!([0], color = :black, label = nothing)
    vline!([-0.5], color = :black, label = nothing)

    min_y = 0
    max_y = 0

    for (i, (plot_df, label)) in enumerate(iter_series(es_result, labels))
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
    results = [
        JLD2.load_object(snakemake.input["treatment_effect_increasing"])[statistic],
        JLD2.load_object(snakemake.input["treatment_effect_decreasing"])[statistic],
    ]
    event_study_plot!(results, ["Increasing Concentration", "Decreasing Concentration"])
end
