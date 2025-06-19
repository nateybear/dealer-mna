using StatsBase: mean, weights as w, median
using Distributions: Normal, pdf
using Dates: Day
using Lazy: @>
using DataFrames
using PartialFunctions
using Missings
using FixedEffectModels

const STATISTICS = (
    mean_baseprice = :mean_price => mean,
    sales_baseprice = :mean_price => sum,
    median_markup = :price_normed => median,
    total_sales = :vehsalesprice => sum,
    total_units = :vehsalesprice => length,
    mean_price = :vehsalesprice => mean,
    trim_rank = :trim_rank => mean,
)

function event_study(
    treated_before,
    treated_after,
    control_before,
    control_after,
    dealer_demographics,
)
    distance = make_distance_fn(dealer_demographics)
    treated_uid = only(unique(treated_before.dealer_uid))
    treated_demographics = only(
        eachrow(subset(dealer_demographics, :uid => ByRow(==(treated_uid))))
    )
    control_uids = unique(control_before.dealer_uid)
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

    all_registrations = vcat(
        treated_before,
        treated_after,
        control_before,
        control_after,
    )

    make_panel((outcol, (incol, f))) = @> all_registrations begin
        groupby([:T, :dealer_uid])
        combine(incol => f => outcol)
    end

    dt(dealer_uid, T, critical_T) = (dealer_uid == treated_uid) * (T == critical_T)

    function ev_term(critical_T)
        return FunctionTerm(
            dt,
            [Term(:dealer_uid), Term(:T), ConstantTerm(critical_T)],
            :(dt(dealer_uid, T, $critical_T)),
        )
    end

    coef_periods = setdiff(VALID_PERIODS, [0])

    function ev_formula(outcol)
        ev_terms = sum(ev_term.(coef_periods)) # drop first period after b/c of colinearity
        rhs = ev_terms + fe(:T) + fe(:dealer_uid)
        return Term(outcol) ~ rhs
    end

    function te_estimate(panel, (outcol, _))
        panel = innerjoin(panel, weights, on = :dealer_uid)
        reg_result = reg(
            panel,
            ev_formula(outcol),
            Vcov.cluster(:dealer_uid),
            # weights = :weight,
            save = :fe,
        )
        table = DataFrame(coeftable(reg_result))
        extract_T(Name) = parse(Int64, match(r"^.*, ([-\d]+)\)$", Name).captures[1])
        table.T = extract_T.(table.Name)
        fes = fe(reg_result, keepkeys = true)
        fe_treated = first(subset(fes, :dealer_uid => ByRow(==(treated_uid))).fe_dealer_uid)
        fes_control = @> fes begin
            subset(:dealer_uid => ByRow(!=(treated_uid)), :T => ByRow(==(0)))
            innerjoin(control_weights, on = :dealer_uid)
        end
        fe_synthetic = mean(fes_control.fe_dealer_uid, w(fes_control.weight))
        return (
            outcol,
            fe_treated,
            fe_synthetic,
            vcov(reg_result),
            select(
                table, :T, :Estimate => outcol,
                Symbol("Std. Error") => Symbol("$(outcol)_se"),
            ),
        )
    end

    function reducer(
        (acc_extra_stats, acc_coeftable),
        (outcol, fe_treated, fe_synthetic, V, curr_coeftable),
    )
        new_extra_stats = DataFrame(stat = outcol, V = Ref(V); fe_treated, fe_synthetic)
        return (
            vcat(acc_extra_stats, new_extra_stats, cols = :union),
            innerjoin(acc_coeftable, curr_coeftable, on = :T),
        )
    end

    init = (DataFrame(), DataFrame(T = coef_periods))

    return mapreduce(reducer, pairs(STATISTICS); init) do s
        panel = make_panel(s)
        te_estimate(panel, s)
    end
end
