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
parse_treated(reg_uid) = @> reg_uid split("_") getindex(2) UUID
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

# function add_regression_weights(dealer_demographics, group)
#     distance = make_distance_fn(dealer_demographics)
#     treated_uid = onlyunique(parse_treated.(group.reg_uid))
#     treated_demographics = only(
#         eachrow(subset(dealer_demographics, :uid => ByRow(==(treated_uid))))
#     )
#     control_uids = setdiff(group.dealer_uid, [treated_uid])
#     control_demographics = subset(dealer_demographics, :uid => ByRow(in(control_uids)))
#     control_distances = [
#         distance(treated_demographics, d)
#         for d in eachrow(control_demographics)
#     ]

#     control_weights = DataFrame(
#         dealer_uid = control_demographics.uid,
#         weight = pdf.(Normal(), control_distances),
#     )

#     control_weights.weight = control_weights.weight ./ sum(control_weights.weight)

#     weights = vcat(control_weights, DataFrame(dealer_uid = treated_uid, weight = 1))

#     return innerjoin(group, weights, on = :dealer_uid)
# end

function make_panel(registrations, (outcol, (incol, f)))
    @> registrations begin
        groupby([:t, :dealer_uid])
        combine(
            incol => f => outcol,
            :τ => onlyunique => :τ,
            :cstime => onlyunique => :cstime,
            :event_time => onlyunique => :event_time,
        )
    end
end

make_multi_formula(t::Tuple) = t
make_multi_formula(t) = (t,)

weighted_mean(x, weight) = mean(x, w(weight))

extract_name(term::Term) = string(term.sym)
extract_name(term::ConstantTerm) = string(term.n)
extract_name(term::InteractionTerm) = join(extract_name.(term.terms), " & ")

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
    treated_dealers,
    heterogeneity_term = ConstantTerm(1),
    dealer_extras = nothing;
    how_join = :dealer_uid, # how to join dealer_extras to the panel
    statistics = keys(STATISTICS), # which statistics to estimate
)
    dt(dealer_uid, τ, critical_τ) = (dealer_uid in treated_dealers) * (τ == critical_τ)

    function ev_term(critical_τ)
        return FunctionTerm(
            dt,
            [Term(:dealer_uid), Term(:τ), ConstantTerm(critical_τ)],
            :(dt(dealer_uid, τ, $critical_τ)),
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
        controls = [
            :median_income,
            :share_urban,
            :closest_first,
            :closest_second,
        ]
        controls = mapreduce(Term, +, controls)
        rhs = ev_terms + fe(:t) * fe(:manufacturer) + fe(:dealer_uid) + fe(:msa)
        return Term(outcol) ~ rhs
    end

    function te_estimate(statistic)
        (outcol, _) = statistic
        panel = make_panel(registrations, statistic)
        if !isnothing(dealer_extras)
            leftjoin!(panel, dealer_extras, on = how_join)
            fillcols!(panel, heterogeneity_term)
        end
        leftjoin!(panel, dealer_demographics, on = :dealer_uid => :uid)
        reg_result = reg(
            panel,
            ev_formula(outcol),
            Vcov.cluster(:dealer_uid),
            save = :fe,
        )
        table = DataFrame(coeftable(reg_result))
        function extract_τ(Name)
            return parse(
                Int64, match(r"dt\(dealer_uid, τ, ([-\d]+)\)", Name).captures[1]
            )
        end
        table.τ = extract_τ.(table.Name)
        function extract_interaction(Name)
            if match(r" & ", Name) !== nothing
                match(r"^dt\(dealer_uid, τ, [-\d]+\) & (.*)$", Name).captures[1]
            else
                "1"
            end
        end
        table.interaction = extract_interaction.(table.Name)
        fes = fe(reg_result, keepkeys = true)
        # fes_treated = @> fes begin
        #     subset([:reg_uid, :dealer_uid] => ByRow(is_treated))
        #     groupby(:reg_uid)
        #     combine(Symbol("fe_dealer_uid&fe_reg_uid") => onlyunique => :fe_treated)
        # end
        # fes_control = @> fes begin
        #     subset([:reg_uid, :dealer_uid] => ByRow(!is_treated))
        #     innerjoin(select(panel, :dealer_uid, :weight), on = :dealer_uid)
        #     groupby(:reg_uid)
        #     combine(
        #         [Symbol("fe_dealer_uid&fe_reg_uid"), :weight] =>
        #             weighted_mean => :fe_synthetic,
        #     )
        # end
        diff_in_diffs = @> fes begin
            innerjoin(panel, on = [:dealer_uid, :t])
            transform(
                [outcol, :fe_dealer_uid] => ByRow(-) => :residual,
                :dealer_uid => ByRow(in(treated_dealers)) => :treated,
            )
            groupby([:t, :treated])
            combine(:residual => mean => :residual)
            sort(:treated, rev = true)
            groupby(:t)
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

    return mapreduce(te_estimate, reducer, pairs(STATISTICS[statistics]); init)
end
