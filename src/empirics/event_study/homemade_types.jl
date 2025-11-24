### ALLOW DIFFERENT TIME TRENDS ###
abstract type AbstractTimeFixedEffect end
struct MetroFixedEffect <: AbstractTimeFixedEffect end
struct ManufacturerFixedEffect <: AbstractTimeFixedEffect end
struct MetroManufacturerFixedEffect <: AbstractTimeFixedEffect end
struct NormalFixedEffect <: AbstractTimeFixedEffect end

fes(::MetroFixedEffect) = fe(:t) & fe(:msa) + fe(:dealer_uid)
fes(::ManufacturerFixedEffect) = fe(:t) & fe(:manufacturer) + fe(:dealer_uid)
function fes(::MetroManufacturerFixedEffect)
    return fe(:t) & fe(:msa) & fe(:manufacturer) + fe(:dealer_uid)
end
fes(::NormalFixedEffect) = fe(:t) + fe(:dealer_uid)


### ALLOW EVENT STUDY INTERACTIONS ###
abstract type AbstractInteractionStyle end
struct NoInteraction <: AbstractInteractionStyle end
struct WithIncludedInteraction{T <: AbstractTerm} <: AbstractInteractionStyle
    term::T
end
@kwdef struct WithAdditionalInteraction{T <: AbstractTerm, THowJoin = Symbol} <: AbstractInteractionStyle
    term::T
    additional_data::DataFrame
    how_join::THowJoin = :dealer_uid
end
const WithInteractionStyle = Union{WithIncludedInteraction, WithAdditionalInteraction}
const NoAdditionalData = Union{NoInteraction, WithIncludedInteraction}

dt(D, τ, critical_τ) = D * (τ == critical_τ)

# dummy for being treated at time τ
function ev_term(critical_τ)
    return FunctionTerm(
        dt,
        [Term(:D), Term(:τ), ConstantTerm(critical_τ)],
        :(dt(D, τ, $critical_τ)),
    )
end

_make_multi_formula(t::Tuple) = t
_make_multi_formula(t) = (t,)

event_terms(::NoInteraction) = mapreduce(ev_term, +, DEFAULT_PERIODS)

import Base.Iterators: product
function event_terms(interaction::WithInteractionStyle)
    all_interactions = product(DEFAULT_PERIODS, _make_multi_formula(interaction.term))
    mapreduce(+, all_interactions) do critical_τ, term
        return ev_term(critical_τ) & term
    end
end

attachadditionaldata!(_, ::NoAdditionalData) = nothing
function attachadditionaldata!(panel, interaction::WithAdditionalInteraction)
    leftjoin!(panel, interaction.additional_data, on = interaction.how_join)
    return nothing
end

function joinadditional!(panel, es::DIYEventStudy)
    leftjoin!(panel, es.dealer_demographics, on = :dealer_uid => :uid)
    attachadditionaldata!(panel, es.interaction)
    attachregressionweights!(panel, dealer_demographics, es.covariate_style, es.covariates)
    return nothing
end


### HOW TO INCLUDE CONTROLS ###
abstract type AbstractCovariateStyle end
struct NoCovariates <: AbstractCovariateStyle end
struct WithRegressionCovariates <: AbstractCovariateStyle end
struct WithRegressionWeights <: AbstractCovariateStyle end

using Folds

attachregressionweights!(_, _, ::Union{NoCovariates,WithRegressionCovariates}, _) = nothing
function attachregressionweights!(panel, demographics, ::WithRegressionWeights, covariates)
    treatment = @> panel begin
        groupby([:dealer_uid, :manufacturer])
        combine(:D => any => :treated)
        innerjoin(panel, on = [:dealer_uid, :manufacturer])
        select(:dealer_uid, :manufacturer, :t, :treated)
    end

    distance = make_distance_fn(demographics, covariates)

    Folds.foreach(groupby(treatment, [:manufacturer, :t]), ThreadedEx(simd=true)) do subdf
        # use views to mutate in place
        treated = subset(subdf, :treated => ByRow(identity), view = true)
        control = subset(subdf, :treated => ByRow(!), view = true)
        treated.weight .= 1.0
        control.weight .= 0.0
        for t in eachrow(treated)
            weights = eachrow(control) .|> distance$(Ref(t)) .|> pdf$(Normal())
            control.weight .+= weights ./ sum(weights)
        end
    end

    select!(treatment, Not(:treated))
    leftjoin!(panel, treatment, on = [:dealer_uid, :manufacturer, :t])
    return nothing
end

add_controls(::Union{NoCovariates, WithRegressionWeights}, _, formula) = formula
function add_controls(::WithRegressionCovariates, covariates, formula)
    controls = mapreduce(Term, +, covariates)
    return formula + controls
end

weight_col(::Union{NoCovariates, WithRegressionCovariates}) = nothing
weight_col(::WithRegressionWeights) = :weight
