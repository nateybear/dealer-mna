using StatsBase: mean, weights as w, median
using Distributions: Normal, pdf
using Dates: Day
using Lazy: @>
using DataFrames
using PartialFunctions
using Missings

const STATISTICS = (
    mean_baseprice = :mean_price => mean,
    sales_baseprice = :mean_price => sum,
    median_markup = :price_normed => median,
    total_sales = :vehsalesprice => sum,
    total_units = :vehsalesprice => length,
    mean_price = :vehsalesprice => mean,
    trim_rank = :trim_rank => mean,
)

const BOOTSTRAP_REPLICAS = 250

function partial_out(df, outcome_col, by_col)
    weight_sum = @> df groupby(:T) combine(:weight => sum ∘ skipmissing => :weight_sum)
    df = @> df begin
        innerjoin(weight_sum, on = :T)
        transform(
            [:weight, :weight_sum] => ((x, y) -> coalesce.(x ./ y, 1)) => :weight_temp
        )
    end
    @> df begin
        groupby(by_col)
        transform(
            [outcome_col, :weight_temp] => ((x, y) -> x .- mean(x, w(y))) => outcome_col
        )
        select(Not(:weight_temp, :weight_sum))
    end
end

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
    control_uids = unique(control_before.dealer_uid) ∪ unique(control_after.dealer_uid)
    control_demographics = subset(dealer_demographics, :uid => ByRow(in(control_uids)))
    control_distances = [
        distance(treated_demographics, d)
        for d in eachrow(control_demographics)
    ]

    control_weights = DataFrame(
        dealer_uid = control_demographics.uid,
        weight = pdf.(Normal(), control_distances),
    )

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

    function shuffle(panel)
        treated_outcome = subset(panel, :dealer_uid => ByRow(==(treated_uid)))
        control_outcomes = subset(panel, :dealer_uid => ByRow(!=(treated_uid)))
        uids = unique(control_outcomes.dealer_uid)
        uids_resample = rand(uids, length(uids))
        control_outcomes = innerjoin(
            control_outcomes,
            DataFrame(dealer_uid = uids_resample),
            on = :dealer_uid,
        )
        return vcat(
            treated_outcome,
            control_outcomes,
        )
    end

    function te_estimate(panel, (outcol, _))
        panel = @> panel begin
            leftjoin(control_weights, on = :dealer_uid)
            partial_out(outcol, :T)
            partial_out(outcol, :dealer_uid)
        end
        combine(groupby(panel, :T)) do gdf
            if !any(gdf.dealer_uid .== treated_uid)
                return (; outcol => missing)
            end
            treated_outcome = only(gdf[gdf.dealer_uid .== treated_uid, outcol])
            control_outcomes = subset(gdf, :dealer_uid => ByRow(!=(treated_uid)))
            synthetic_control_outcome = mean(
                control_outcomes[!, outcol],
                w(disallowmissing(control_outcomes.weight)),
            )

            (; outcol => (treated_outcome - synthetic_control_outcome))
        end
    end

    sample_estimate = mapreduce(innerjoin$(on = :T,), pairs(STATISTICS)) do s
        panel = make_panel(s)
        te_estimate(panel, s)
    end

    # B is the bootstrap number (missing means in-sample estimate)
    sample_estimate.B .= missing

    out = sample_estimate

    for i in Base.OneTo(BOOTSTRAP_REPLICAS)
        @info "Bootstrap iteration $(i)"
        estimate = mapreduce(innerjoin$(on = :T,), pairs(STATISTICS)) do s
            panel = make_panel(s)
            te_estimate(shuffle(panel), s)
        end
        estimate.B .= i
        out = vcat(out, estimate)
    end

    return out
end
