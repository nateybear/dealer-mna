include("homemade_types.jl")

@kwdef struct DIYEventStudy{
    TFEs<:AbstractTimeFixedEffect, 
    TInteraction <: AbstractInteractionStyle = NoInteraction,
    TCovariateStyle <: AbstractCovariateStyle = AsRegressionWeights
} <: AbstractEventStudy
    dealer_demographics::DataFrame
    registrations::DataFrame
    covariates::Vector{Symbol} = DEFAULT_DEMOGRAPHICS
    interaction::TInteraction = TInteraction()
    covariate_style::TCovariateStyle = TCovariateStyle()
    fixed_effect_style::TFEs = TFEs()
end

groupcols(::DIYEventStudy{MetroFixedEffect}) = [:t, :dealer_uid, :msa]
groupcols(::DIYEventStudy{ManufacturerFixedEffect}) = [:t, :dealer_uid, :manufacturer]
groupcols(::DIYEventStudy{MetroManufacturerFixedEffect}) = [:t, :dealer_uid, :msa, :manufacturer]

function event_study(es::DIYEventStudy, statistic)
    # create panel
    panel = make_panel(es, statistic)
    joinadditional!(panel, es)

    # run regression
    event_col, _ = statistic
    formula = event_terms(es.interaction) + fes(es.fixed_effect_style)
    formula = add_controls(es.covariate_style, es.covariates, formula)
    formula = Term(event_col) ~ formula
    reg_result = reg(
        panel,
        formula,
        Vcov.cluster(:dealer_uid),
        save = :fe,
        weights = weight_col(es.covariate_style),
    )
end
