# Use the did package from R to do a Callaway-Sant'anna event study
using RCall

import ..Dealers, ..snakemake

struct CSEventStudyConfig <: AbstractEventStudyConfig
    registrations::AbstractDataFrame
    dealer_demographics::DataFrame
    covariates::Vector{Symbol}
end

function CSEventStudyConfig(;
    registrations,
    dealer_demographics,
    covariates = DEFAULT_DEMOGRAPHICS,
)
    transactions = JLD2.load_object(snakemake.input["transactions_for_analysis"])
    cstimetable = flatten(
        DataFrame(
            (
                dealer_uid = Dealers.State.df.uid[Dealers.dealer_idxs(t)],
                event_time = Dealers.cstime(t),
            ) for t in transactions
        ),
        :dealer_uid,
    )
    leftjoin!(registrations, cstimetable, on = :dealer_uid)
    registrations.event_time = coalesce.(registrations.event_time, 0)
    registrations.cstime = Dealers.cstime.(registrations.t)
    return CSEventStudyConfig(
        registrations,
        dealer_demographics,
        covariates,
    )
end

registrations(es::CSEventStudyConfig) = es.registrations
summarycols(::CSEventStudyConfig) = [:τ, :D, :cstime, :event_time]

to_index(xs) = let unique_x = unique(xs)
    map(x -> findfirst(==(x), unique_x), xs)
end

function event_study(es::CSEventStudyConfig, statistic)
    covariates = string.(es.covariates)
    panel = make_panel(es, statistic)
    leftjoin!(panel, es.dealer_demographics, on = :dealer_uid => :uid)
    transform!(panel, :dealer_uid => to_index => :dealer_uid)

    # run event study in R
    event_col, _ = statistic
    event_col = string(event_col)
    @rput covariates panel event_col
    R"""
    library(did)
    estimate <- att_gt(
        yname = event_col,
        tname = "cstime",
        idname = "dealer_uid",
        gname = "event_time",
        data = panel,
        est_method = "dr",
        allow_unbalanced_panel = TRUE,
        anticipation = 1,
    )
    agg <- aggte(estimate, type = "dynamic", na.rm = TRUE, min_e = -4, max_e = 3)

    # reduce size of object and make serializable for Julia
    agg$call <- NULL
    agg$DIDparams <- NULL
    """
    @rget agg
    return EventStudyResult(
        Tuple(Int8.(agg[:egt])),
        Tuple(agg[:att_egt]),
        Tuple(agg[:se_egt]),
        agg[:overall_att],
        agg[:overall_se],
        nothing
    )
end
