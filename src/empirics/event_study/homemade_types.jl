### ALLOW DIFFERENT TIME TRENDS ###
abstract type AbstractTimeFixedEffect end
struct MetroFixedEffect <: AbstractTimeFixedEffect end
struct ManufacturerFixedEffect <: AbstractTimeFixedEffect end
struct MetroManufacturerFixedEffect <: AbstractTimeFixedEffect end
struct NormalFixedEffect <: AbstractTimeFixedEffect end
struct SyntheticPairFixedEffect <: AbstractTimeFixedEffect end

### CONTROL SUN & ABRAHAM AGGREGATION ###
abstract type AbstractAggregationStyle end
struct SunAbraham <: AbstractAggregationStyle end
struct TwoWayFixedEffects <: AbstractAggregationStyle end

fes(::MetroFixedEffect) = fe(:t) & fe(:msa) + fe(:dealer_uid)
fes(::ManufacturerFixedEffect) = fe(:t) & fe(:manufacturer) + fe(:dealer_uid)
fes(::NormalFixedEffect) = fe(:t) + fe(:dealer_uid)
fes(::MetroManufacturerFixedEffect) = fe(:t) & fe(:msa) + fe(:manufacturer) + fe(:dealer_uid)
fes(::SyntheticPairFixedEffect) = fe(:t) & fe(:manufacturer) + fe(:pair_id)

### ALLOW EVENT STUDY INTERACTIONS ###
abstract type AbstractInteractionStyle end
struct NoInteraction <: AbstractInteractionStyle end
struct WithIncludedInteraction{T <: Union{AbstractTerm, Tuple{Vararg{<:AbstractTerm}}}} <:
       AbstractInteractionStyle
    term::T
end
@kwdef struct WithAdditionalInteraction{
    T <: Union{AbstractTerm, Tuple{Vararg{<:AbstractTerm}}}, THowJoin
} <: AbstractInteractionStyle
    term::T
    additional_data::AbstractDataFrame
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

function cohort_term(critical_cohort)
    return FunctionTerm(
        cohort -> cohort == critical_cohort,
        [Term(:cohort)],
        :(cohort == $critical_cohort),
    )
end

_make_multi_formula(t::Tuple) = t
_make_multi_formula(t) = (t,)

import Base.Iterators: product
function event_terms(::NoInteraction, cohorts, ::SunAbraham)
    all_cohorts = product(cohorts, DEFAULT_PERIODS)
    mapreduce(+, all_cohorts) do (critical_cohort, critical_τ)
        return cohort_term(critical_cohort) & ev_term(critical_τ)
    end
end

function event_terms(interaction::WithInteractionStyle, cohorts, ::SunAbraham)
    all_interactions = product(cohorts, DEFAULT_PERIODS, _make_multi_formula(interaction.term))
    mapreduce(+, all_interactions) do (critical_cohort, critical_τ, term)
        return cohort_term(critical_cohort) & ev_term(critical_τ) & term
    end
end

event_terms(::NoInteraction, _, ::TwoWayFixedEffects) =
    mapreduce(+, DEFAULT_PERIODS) do critical_τ
        return ev_term(critical_τ)
    end

function event_terms(interaction::WithInteractionStyle, _, ::TwoWayFixedEffects)
    all_interactions = product(DEFAULT_PERIODS, _make_multi_formula(interaction.term))
    mapreduce(+, all_interactions) do (critical_τ, term)
        return ev_term(critical_τ) & term
    end
end

df_cols(t::Tuple) = unique(Iterators.flatten(df_cols.(t)))
df_cols(t::InteractionTerm) = df_cols(t.terms)
df_cols(t::Term) = [t.sym]
df_cols(::ConstantTerm) = []

# fill unsupplied dealer cols with 0 so FixedEffectModels doesn't drop them from regression
function fillcols!(df, form)
    cols = df_cols(form)
    df[!, cols] .= coalesce.(df[!, cols], 0)
    return nothing
end

attachadditionaldata!(_, ::NoAdditionalData) = nothing
function attachadditionaldata!(panel, interaction::WithAdditionalInteraction)
    leftjoin!(panel, interaction.additional_data, on = interaction.how_join)
    fillcols!(panel, interaction.term)
    return nothing
end

### HOW TO INCLUDE CONTROLS ###
abstract type AbstractCovariateStyle end
struct NoCovariates <: AbstractCovariateStyle end
@kwdef struct WithRegressionCovariates <: AbstractCovariateStyle
    covariates::Vector{Symbol} = DEFAULT_DEMOGRAPHICS
end
@kwdef struct WithRegressionWeights <: AbstractCovariateStyle
    covariates::Vector{Symbol} = DEFAULT_DEMOGRAPHICS
end

function create_manufacturer_weights!(panel)
    all_counts = combine(groupby(panel, [:dealer_uid, :manufacturer]), :n => sum)
    weights = transform(groupby(all_counts, :dealer_uid), :n_sum => (x -> x ./ sum(x)) => :weight)
    select!(weights, Not(:n_sum))
    leftjoin!(panel, weights, on = [:dealer_uid, :manufacturer])
    return nothing
end

# re-weight regression weights for when there are two manufacturers at one dealership
function attachregressionweights!(panel, _, ::Union{NoCovariates, WithRegressionCovariates})
    create_manufacturer_weights!(panel)
    return nothing
end
function attachregressionweights!(panel, demographics, w::WithRegressionWeights)
    create_manufacturer_weights!(panel)
    create_synthetic_pairs!(panel, demographics, w.covariates)
    return nothing
end

# Create synthetic control pairs following Brand et al. approach
function create_synthetic_pairs!(panel, demographics, covariates)
    distance = make_distance_fn(demographics, covariates)

    # Separate treated and control observations
    control_panel = DataFrame()

    for group in groupby(panel, [:t, :manufacturer])
        treated_rows = subset(group, :D, view = true)
        control_rows = subset(group, :D => ByRow(!))

        treated_rows.pair_id = string.(treated_rows.dealer_uid)

        if isempty(control_rows)
            @error "No control dealers available for time $(group.t[1]) and manufacturer $(group.manufacturer[1])"
            continue
        end

        for t in eachrow(treated_rows)
            distances = [distance(t, c) for c in eachrow(control_rows)]
            weights = pdf.(Normal(), distances)
            weights = weights ./ sum(weights)
            new_controls = deepcopy(control_rows)
            new_controls.pair_id .= t.pair_id
            new_controls.weight .= weights .* t.weight
            append!(control_panel, new_controls)
        end
    end

    subset!(panel, :D)
    append!(panel, control_panel)

    return nothing
end

add_controls(::Union{NoCovariates, WithRegressionWeights}, formula) = formula
function add_controls(w::WithRegressionCovariates, formula)
    controls = mapreduce(Term, +, w.covariates)
    return formula + controls
end

### SHAPING EVENT STUDY RESULTS ###

extract_τ(Name) = parse(Int64, match(r"dt\(D, τ, ([-\d]+)\)", Name).captures[1])

extract_cohort(Name) =
    parse(Date, match(r"cohort == Date\(\"(.*)\"\) & dt", Name).captures[1])

extract_interaction(Name) =
    if !isnothing(match(r"dt\(.*\) & .*$", Name))
        match(r"dt\(D, τ, [-\d]+\) & (.*)$", Name).captures[1]
    else
        "1"
    end

weighted_mean(vals, weights) = sum(vals .* weights) / sum(weights)

function collect_dids(reg_result, panel)
    treated = subset(panel, :D)
    actual_treated = treated[!, reg_result.responsename]
    treated.D .= 0
    pred_control = predict(reg_result, treated)
    treated.residual = actual_treated .- pred_control

    return @> treated begin
        subset(:τ => ByRow(in((-4, -3, 2, 3)))) # only need to difference residuals of treated dealers
        transform(:τ => ByRow(sign), renamecols = false)
        groupby([:dealer_uid, :manufacturer, :τ])
        combine(
            [:residual, :weight] => weighted_mean => :residual,
            :weight => sum => :weight,
        )
        transform(:τ => ByRow(t -> t > 0 ? "after" : "before") => :period)
        unstack([:dealer_uid, :manufacturer, :weight], :period, :residual, fill = 0)
        transform([:after, :before] => (-) => :diff_in_diff)
        groupby(:dealer_uid)
        combine(
            ([col, :weight] => weighted_mean => col for col in [:before, :after, :diff_in_diff])...,
            renamecols = false,
        )
    end
end

function aggregate_table_interaction(panel, reg_result, ::TwoWayFixedEffects)
    table = DataFrame(coeftable(reg_result))
    table.τ = extract_τ.(table.Name)
    table.interaction = extract_interaction.(table.Name)

    estimates = table.Estimate
    V_direct = vcov(reg_result)

    interactions = unique(table.interaction)

    return table, interactions, estimates, V_direct
end

function aggregate_table_interaction(panel, reg_result, ::SunAbraham)
    table = DataFrame(coeftable(reg_result))
    table.τ = extract_τ.(table.Name)
    table.cohort = extract_cohort.(table.Name)
    table.interaction = extract_interaction.(table.Name)
    nan_mask = .!isnan.(table[!, "Std. Error"])
    subset!(table, Symbol("Std. Error") => ByRow(!isnan)) # some excluded terms

    cohort_props = @> panel begin
        subset(:cohort => ByRow(>(Date(2000))))
        groupby(:cohort)
        combine(proprow => :weight)
    end

    # rebalance because of excluded terms
    table = combine(groupby(table, [:τ, :interaction])) do subdf
        out = innerjoin(subdf, cohort_props, on = :cohort)
        out.weight .= out.weight ./ sum(out.weight)
        return out
    end

    gtable = groupby(table, [:τ, :interaction])

    group_mask = let c = reshape(unique(groupindices(gtable)), 1, :)
        (groupindices(gtable) .== c) .* table.weight
    end

    estimates = group_mask'table.Estimate
    V_agg = group_mask' * vcov(reg_result)[nan_mask, nan_mask] * group_mask

    table = combine(gtable, nrow)
    interactions = unique(table.interaction)

    return table, interactions, estimates, V_agg
end

function EventStudyResult(
    panel::DataFrame,
    reg_result::FixedEffectModel,
    ::WithInteractionStyle,
    aggregation_style,
)
    table, interactions, estimates, V_agg = aggregate_table_interaction(
        panel, reg_result, aggregation_style
    )

    ∇(i) =
        let c = reshape(interactions, 1, :)
            (table.τ .== i) .* (table.interaction .== c)
        end

    est(nabla) = NamedArray(
        nabla' * estimates,
        (interactions,),
    )

    V(nabla) = NamedArray(
        nabla' * V_agg * nabla,
        (interactions, interactions),
    )

    did_nabla = sum(∇.(2:3)) / 2 - sum(∇.(-4:-3)) / 2

    return EventStudyResult(
        estimates = Tuple(est.(∇.(DEFAULT_PERIODS))),
        std_errs = Tuple(V.(∇.(DEFAULT_PERIODS))),
        did_estimate = est(did_nabla),
        did_stderr = V(did_nabla),
        extras = collect_dids(reg_result, panel),
    )
end

function aggregate_table_no_interaction(panel, reg_result, ::TwoWayFixedEffects)
    table = DataFrame(coeftable(reg_result))
    table.τ = extract_τ.(table.Name)

    estimates = table.Estimate
    V = vcov(reg_result)

    return table, estimates, V
end

function aggregate_table_no_interaction(panel, reg_result, ::SunAbraham)
    table = DataFrame(coeftable(reg_result))
    table.τ = extract_τ.(table.Name)
    table.cohort = extract_cohort.(table.Name)
    nan_mask = .!isnan.(table[!, "Std. Error"])
    subset!(table, Symbol("Std. Error") => ByRow(!isnan)) # some excluded terms

    cohort_props = @> panel begin
        subset(:cohort => ByRow(>(Date(2000))))
        groupby(:cohort)
        combine(proprow => :weight)
    end

    # rebalance because of excluded terms
    table = combine(groupby(table, :τ)) do subdf
        out = innerjoin(subdf, cohort_props, on = :cohort)
        out.weight .= out.weight ./ sum(out.weight)
        return out
    end

    group_mask = let c = reshape(unique(table.τ), 1, :)
        (table.τ .== c) .* table.weight
    end

    estimates = group_mask'table.Estimate
    V_agg = group_mask' * vcov(reg_result)[nan_mask, nan_mask] * group_mask

    return table, estimates, V_agg
end

function EventStudyResult(
    panel::DataFrame,
    reg_result::FixedEffectModel,
    ::NoInteraction,
    aggregation_style,
)
    table, estimates, V = aggregate_table_no_interaction(panel, reg_result, aggregation_style)

    ∇ = [-1 / 2, -1 / 2, 0, 0, 0, 1 / 2, 1 / 2]
    difference = ∇'estimates
    se_difference = sqrt(∇' * V * ∇)

    return EventStudyResult(
        estimates = Tuple(estimates),
        std_errs = Tuple(sqrt.(diag(V))),
        did_estimate = difference,
        did_stderr = se_difference,
        extras = collect_dids(reg_result, panel),
    )
end

# Fallback methods for backward compatibility (default to SunAbraham)
function EventStudyResult(panel, reg_result, interaction_style)
    return EventStudyResult(panel, reg_result, interaction_style, SunAbraham())
end
