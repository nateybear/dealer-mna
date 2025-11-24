using StatsBase: mean, weights as w, median
using Distributions: Normal, pdf
using Dates: Day
using Lazy: @>
using DataFrames
using PartialFunctions
using Missings
using FixedEffectModels

const DEFAULT_PERIODS = setdiff(VALID_PERIODS, [0])
const DEFAULT_DEMOGRAPHICS = [
    :total_population,
    :median_income,
    :share_urban,
    :share_black,
    :share_hispanic,
]
const STATISTICS = (
    mean_baseprice = :base_price => mean,
    sales_baseprice = :base_price => sum,
    median_markup = :price_normed => median,
    total_sales = :vehsalesprice => sum,
    total_units = :vehsalesprice => length,
    mean_price = :vehsalesprice => mean,
    trim_rank = :trim_rank => mean,
    unique_trims = [:model, :extra] => numunique,
)

abstract type AbstractEventStudy end

registrations(::AbstractEventStudy) = error("unimplemented")
groupcols(::AbstractEventStudy) = [:t, :dealer_uid]

@kwdef struct EventStudyResult{
    N,
    TExtra = Nothing,
    TEstimate = Float64,
    TDidEstimate = Float64,
}
    periods::NTuple{N, Int8} = DEFAULT_PERIODS
    estimates::NTuple{N, TEstimate}
    std_errs::NTuple{N, TEstimate}
    did_estimate::TDidEstimate
    did_stderr::TDidEstimate
    extras::TExtra = nothing
end

function make_panel(event_study, (outcol, (incol, f)))
    @> registrations(event_study) begin
        groupby(groupcols(event_study))
        combine(
            incol => f => outcol,
            [:τ, :D, :cstime, :event_time] .=> onlyunique,
            renamecols = false,
        )
    end
end
