using StatsBase: mean, weights as w, median
using Distributions: Normal, pdf
using Dates: Day
using Lazy: @>
using DataFrames
using PartialFunctions
using Missings
using FixedEffectModels

numunique(cols...) = cols |> splat(zip) |> unique |> length
onlyunique = only ∘ unique
parse_treated(reg_uid) = @> reg_uid split("_") last UUID
is_treated(reg_uid, dealer_uid) = dealer_uid == parse_treated(reg_uid)

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

function add_regression_weights(dealer_demographics, group)
    distance = make_distance_fn(dealer_demographics)
    treated_uid = onlyunique(parse_treated.(group.reg_uid))
    treated_demographics = only(
        eachrow(subset(dealer_demographics, :uid => ByRow(==(treated_uid))))
    )
    control_uids = setdiff(group.dealer_uid, [treated_uid])
    control_demographics = subset(dealer_demographics, :uid => ByRow(in(control_uids)))
    control_distances = [
        distance(treated_demographics, d)
        for d in eachrow(control_demographics)
    ]

    control_weights = DataFrame(
        dealer_uid = control_demographics.uid,
        weight = pdf.(Normal(), control_distances),
    )

    control_weights.weight = control_weights.weight ./ sum(control_weights.weight)

    weights = vcat(control_weights, DataFrame(dealer_uid = treated_uid, weight = 1))

    return innerjoin(group, weights, on = :dealer_uid)
end

make_panel(registrations, (outcol, (incol, f))) = @> registrations begin
    groupby([:T, :dealer_uid, :reg_uid])
    combine(incol => f => outcol)
end

make_multi_formula(t::Tuple) = t
make_multi_formula(t) = (t,)

weighted_mean(x, weight) = mean(x, w(weight))

extract_name(term::Term) = string(term.sym)
extract_name(term::ConstantTerm) = string(term.n)
extract_name(term::InteractionTerm) = join(extract_name.(term.terms), "&")

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

function event_study(
    registrations,
    dealer_demographics,
    heterogeneity_term = ConstantTerm(1),
    dealer_extras = nothing,
    how_join = :dealer_uid, # how to join dealer_extras to the panel
)
    function dt(dealer_uid, reg_uid, T, critical_T)
        return (dealer_uid == parse_treated(reg_uid)) * (T == critical_T)
    end

    function ev_term(critical_T)
        return FunctionTerm(
            dt,
            [Term(:dealer_uid), Term(:reg_uid), Term(:T), ConstantTerm(critical_T)],
            :(dt(dealer_uid, reg_uid, T, $critical_T)),
        )
    end

    # drop first period after b/c of colinearity
    coef_periods = setdiff(VALID_PERIODS, [0])

    function ev_formula(outcol)
        ev_terms = mapreduce(
            splat(&),
            +,
            Iterators.product(
                ev_term.(coef_periods), make_multi_formula(heterogeneity_term)
            ),
        )
        rhs = ev_terms + fe(:T) & fe(:reg_uid) + fe(:dealer_uid) & fe(:reg_uid)
        return Term(outcol) ~ rhs
    end

    function te_estimate(statistic)
        (outcol, _) = statistic
        panel = @> registrations begin
            make_panel(statistic)
            groupby(:reg_uid)
            combine(add_regression_weights$dealer_demographics)
        end
        if !isnothing(dealer_extras)
            leftjoin!(panel, dealer_extras, on = how_join)
            fillcols!(panel, heterogeneity_term)
        end
        reg_result = reg(
            panel,
            ev_formula(outcol),
            Vcov.cluster(:dealer_uid),
            weights = :weight,
            save = :fe,
        )
        table = DataFrame(coeftable(reg_result))
        function extract_T(Name)
            return parse(
                Int64, match(r"dt\(dealer_uid, reg_uid, T, ([-\d]+)\)", Name).captures[1]
            )
        end
        table.T = extract_T.(table.Name)
        extract_interaction(Name) =
            if match(r" & ", Name) !== nothing
                match(r" & (.*)$", Name).captures[1]
            else
                "1"
            end
        table.interaction = extract_interaction.(table.Name)
        fes = fe(reg_result, keepkeys = true)
        fes_treated = @> fes begin
            subset([:reg_uid, :dealer_uid] => ByRow(is_treated))
            groupby(:reg_uid)
            combine(Symbol("fe_dealer_uid&fe_reg_uid") => onlyunique => :fe_treated)
        end
        fes_control = @> fes begin
            subset([:reg_uid, :dealer_uid] => ByRow(!is_treated))
            innerjoin(select(panel, :dealer_uid, :weight), on = :dealer_uid)
            groupby(:reg_uid)
            combine(
                [Symbol("fe_dealer_uid&fe_reg_uid"), :weight] =>
                    weighted_mean => :fe_synthetic,
            )
        end
        diff_in_diffs = @> fes begin
            innerjoin(panel, on = [:reg_uid, :dealer_uid, :T])
            transform(
                [outcol, Symbol("fe_dealer_uid&fe_reg_uid")] => ByRow(-) => :residual,
                [:reg_uid, :dealer_uid] => ByRow(is_treated) => :treated,
            )
            groupby([:T, :reg_uid, :treated])
            combine([:residual, :weight] => weighted_mean => :residual)
            sort(:treated, rev = true)
            groupby([:T, :reg_uid])
            combine(:residual => splat(-) => :diff)
            subset(:T => ByRow(in((-3, 2))))
            sort(:T, rev = true)
            groupby(:reg_uid)
            combine(:diff => splat(-) => :diff_in_diff)
        end
        fes = @> fes_treated begin
            innerjoin(fes_control, on = :reg_uid)
            innerjoin(diff_in_diffs, on = :reg_uid)
        end
        return (
            outcol,
            fes,
            vcov(reg_result),
            select(
                table, :T, :interaction, :Estimate => outcol,
                Symbol("Std. Error") => Symbol("$(outcol)_se"),
            ),
        )
    end

    function reducer(
        (acc_extra_stats, acc_coeftable),
        (outcol, fes, V, curr_coeftable),
    )
        new_extra_stats = DataFrame(stat = outcol, V = Ref(V), fes = Ref(fes))
        return (
            vcat(acc_extra_stats, new_extra_stats, cols = :union),
            innerjoin(acc_coeftable, curr_coeftable, on = [:T, :interaction]),
        )
    end

    init = (
        DataFrame(),
        allcombinations(
            DataFrame,
            :T => coef_periods,
            :interaction =>
                collect(map(extract_name, make_multi_formula(heterogeneity_term))),
        ),
    )

    return mapreduce(te_estimate, reducer, pairs(STATISTICS); init)
end
