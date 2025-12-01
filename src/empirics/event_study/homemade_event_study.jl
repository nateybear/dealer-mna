@kwdef struct DIYEventStudyConfig{
    TFEs <: AbstractTimeFixedEffect,
    TInteraction <: AbstractInteractionStyle,
    TCovariateStyle <: AbstractCovariateStyle,
} <: AbstractEventStudyConfig
    registrations::AbstractDataFrame
    dealer_demographics::DataFrame
    interaction_style::TInteraction = NoInteraction()
    covariate_style::TCovariateStyle = WithRegressionWeights()
    fixed_effect_style::TFEs = ManufacturerFixedEffect()
end

registrations(es::DIYEventStudyConfig) = es.registrations
function registrations(
    es::DIYEventStudyConfig{<:Union{MetroFixedEffect, MetroManufacturerFixedEffect}}
)
    out = innerjoin(
        es.registrations,
        select(Dealers.State.df, :uid, :msa),
        on = :dealer_uid => :uid,
    )
    out.msa .= something.(out.msa, "OUTSIDE MSA")
    return out
end

groupcols(::DIYEventStudyConfig{MetroFixedEffect}) = [:t, :dealer_uid, :msa, :manufacturer]
groupcols(::DIYEventStudyConfig{NormalFixedEffect}) = [:t, :dealer_uid, :manufacturer]
groupcols(::DIYEventStudyConfig{ManufacturerFixedEffect}) = [:t, :dealer_uid, :manufacturer]
function groupcols(::DIYEventStudyConfig{MetroManufacturerFixedEffect})
    return [:t, :dealer_uid, :msa, :manufacturer]
end

function joinadditional!(panel, es::DIYEventStudyConfig)
    leftjoin!(panel, es.dealer_demographics, on = :dealer_uid => :uid)
    attachadditionaldata!(panel, es.interaction_style)
    attachregressionweights!(panel, es.dealer_demographics, es.covariate_style)
    addcohorts!(panel)
    return nothing
end

function addcohorts!(panel)
    cohort_join = subset(panel, :τ => ByRow(==( -4)))
    select!(cohort_join, :dealer_uid, :t => :cohort)
    unique!(cohort_join)
    leftjoin!(panel, cohort_join, on = :dealer_uid)
    panel.cohort .= coalesce.(panel.cohort, Date(1900))
    return nothing
end

function event_study(es::DIYEventStudyConfig, statistic)
    # create panel
    panel = make_panel(es, statistic)
    joinadditional!(panel, es)

    cohorts = [ d for d in unique(panel.cohort) if d > Date(2000) ]

    # run regression
    event_col, _ = statistic
    formula = event_terms(es.interaction_style, cohorts) + fes(es.fixed_effect_style)
    formula = add_controls(es.covariate_style, formula)
    formula = Term(event_col) ~ formula
    reg_result = reg(
        panel,
        formula,
        Vcov.cluster(:dealer_uid),
        save = :fe,
        weights = :weight,
    )
    return EventStudyResult(
        panel,
        reg_result,
        es.interaction_style,
    )
end
