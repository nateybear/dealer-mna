using StatsBase: mean, weights as w, median, cov
using Distributions: Normal, pdf, MvNormal
using Dates: Day
using Lazy: @>
using DataFrames
using PartialFunctions
using Missings
using FixedEffectModels

numunique(cols...) = cols |> splat(zip) |> unique |> length

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

function event_study(
    treated_before,
    treated_after,
    control_before,
    control_after,
    dealer_demographics,
)
    distance = make_distance_fn(dealer_demographics)

    all_uids = treated_before.dealer_uid ∪ control_after.dealer_uid

    function make_weights(treated_uid)
        treated_demographics = only(
            eachrow(subset(dealer_demographics, :uid => ByRow(==(treated_uid))))
        )
        control_uids = setdiff(all_uids, [treated_uid])
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

        return vcat(control_weights, DataFrame(dealer_uid = treated_uid, weight = 1))
    end

    all_registrations = vcat(
        treated_before,
        treated_after,
        control_before,
        control_after,
        cols = :union,
    )

    make_panel((outcol, (incol, f))) = @> all_registrations begin
        groupby([:T, :dealer_uid])
        combine(incol => f => outcol)
    end

    function dt(treated_uid, dealer_uid, T, critical_T)
        return (dealer_uid == treated_uid) * (T == critical_T)
    end

    function ev_term(treated_uid, critical_T)
        return FunctionTerm(
            dt$(treated_uid),
            [Term(:dealer_uid), Term(:T), ConstantTerm(critical_T)],
            :(dt(dealer_uid, T, $critical_T)),
        )
    end

    coef_periods = setdiff(VALID_PERIODS, [0])

    function ev_formula(treated_uid, outcol)
        ev_terms = sum(ev_term.(Ref(treated_uid), coef_periods)) # drop first period after b/c of colinearity
        rhs = ev_terms + fe(:T) + fe(:dealer_uid)
        return Term(outcol) ~ rhs
    end

    treated_uid = only(unique(treated_before.dealer_uid))
    control_uids = setdiff(all_uids, [treated_uid])

    function te_estimate(panel, (outcol, _))
        weights = make_weights(treated_uid)
        reg_result = reg(
            innerjoin(panel, weights, on = :dealer_uid),
            ev_formula(treated_uid, outcol),
            weights = :weight,
            save = :fe,
        )
        table = DataFrame(coeftable(reg_result))
        extract_T(Name) = parse(Int64, match(r"^.*, ([-\d]+)\)$", Name).captures[1])
        table.T = extract_T.(table.Name)
        fes = fe(reg_result, keepkeys = true)
        fe_treated = first(subset(fes, :dealer_uid => ByRow(==(treated_uid))).fe_dealer_uid)
        fes_control = @> fes begin
            subset(:dealer_uid => ByRow(!=(treated_uid)), :T => ByRow(==(0)))
            innerjoin(weights, on = :dealer_uid)
        end
        fe_synthetic = mean(fes_control.fe_dealer_uid, w(fes_control.weight))

        placebos = mapreduce(vcat, 1:250) do _
            new_treated_uid = rand(control_uids)
            new_weights = make_weights(new_treated_uid)
            coef(
                reg(
                    innerjoin(panel, new_weights, on = :dealer_uid),
                    ev_formula(new_treated_uid, outcol),
                    weights = :weight,
                ),
            )'
        end

        placebo_distn = MvNormal(reshape(mean(placebos, dims = 1), :), cov(placebos))

        return (
            outcol,
            fe_treated,
            fe_synthetic,
            placebo_distn,
            select(
                table, :T, :Estimate => outcol,
                Symbol("Std. Error") => Symbol("$(outcol)_se"),
            ),
        )
    end

    function reducer(
        (acc_extra_stats, acc_coeftable),
        (outcol, fe_treated, fe_synthetic, D_placebo, curr_coeftable),
    )
        new_extra_stats = DataFrame(
            stat = outcol, D_placebo = Ref(D_placebo); fe_treated, fe_synthetic
        )
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
