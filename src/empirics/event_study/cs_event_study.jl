# Use the did package from R to do a Callaway-Sant'anna event study
using RCall

@kwdef struct CSEventStudy
    dealer_demographics::DataFrame
    registrations::DataFrame
    covariates::Vector{Symbol} = DEFAULT_DEMOGRAPHICS
end

registrations(es::CSEventStudy) = es.registrations

function event_study(es::CSEventStudy, statistic)
    covariates = string.(covariates)
    panel = make_panel(es, statistic)
    event_col, _ = statistic
    event_col = string(event_col)
    @rput covariates panel event_col
    R"""
    library(did)
    estimate <- att_gt(
        yname = event_col,
        tname = "cstime",
        idname = "dealer_uid",
        gname = "cs_eventtime",
        data = panel,
        xformla = as.formula(paste("~", paste(covariates, collapse = "+"))),
        est_method = "ipw",
        control_group = "never-treated",
        allow_unbalanced_panel = TRUE,
        anticipation = 1,
    )
    agg <- aggte(estimate, type = "dynamic", min_e = -4, max_e = 3)

    # reduce size of object and make serializable for Julia
    agg$call <- NULL
    agg$DIDparams <- NULL
    """
    @rget agg
    return EventStudyResult(
        agg[:egt],
        agg[:att_egt],
        agg[:se_egt],
        agg[:overall_att],
        agg[:overall_se],
    )
end
