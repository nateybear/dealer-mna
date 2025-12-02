using StatsBase: mean, weights as w, median, mode
using Distributions: Normal, pdf
using Dates: Day
using Lazy: @>
using DataFrames
using PartialFunctions
using Missings
using FixedEffectModels
using NamedArrays

numunique(cols...) = cols |> splat(zip) |> unique |> length
onlyunique = only ∘ unique

const DEFAULT_PERIODS = setdiff(VALID_PERIODS, [-1])
const DEFAULT_DEMOGRAPHICS = [
    :total_population,
    :median_income,
    :share_urban,
    :share_black,
    :share_hispanic,
]
const STATISTICS = (
    mean_baseprice = :base_price => mean,
    median_markup = :price_normed => median,
    total_sales = :vehsalesprice => sum,
    total_units = :vehsalesprice => length,
    mean_price = :vehsalesprice => mean,
    trim_rank = :trim_rank => mean,
    unique_trims = [:model, :extra] => numunique,
)

abstract type AbstractEventStudyConfig end

registrations(::AbstractEventStudyConfig) = error("unimplemented")
groupcols(::AbstractEventStudyConfig) = [:t, :dealer_uid]
summarycols(::AbstractEventStudyConfig) = [:τ, :D]

@kwdef struct EventStudyResult{
    N,
    TExtra,
    TEstimate,
    TStdErr,
}
    periods::NTuple{N, Int8} = Tuple(Int8.(DEFAULT_PERIODS))
    estimates::NTuple{N, TEstimate}
    std_errs::NTuple{N, TStdErr}
    did_estimate::TEstimate
    did_stderr::TStdErr
    extras::TExtra = nothing
end

function make_panel(es, statistic)
    (outcol, (incol, f)) = statistic
    @> registrations(es) begin
        groupby(groupcols(es))
        combine(
            incol => f => outcol,
            summarycols(es) .=> onlyunique,
            nrow => :n,
            renamecols = false,
        )
    end
end

include("cs_event_study.jl")
include("homemade_types.jl")
include("homemade_event_study.jl")

function event_study(es::AbstractEventStudyConfig)
    results = Dict{Symbol, EventStudyResult}()
    for statistic in pairs(STATISTICS)
        stat_name, _ = statistic
        results[stat_name] = event_study(es, statistic)
    end
    return results
end

function parse_event_study_config(str)
    if str == "callawaysantanna"
        CSEventStudyConfig
    elseif str == "weights"
        DIEventStudyConfig$(covariate_spec = WithRegressionWeights(),)
    elseif str == "controls"
        DIEventStudyConfig$(covariate_spec = WithRegressionCovariates(),)
    else
        error("Unknown event study specification: $str")
    end
end
