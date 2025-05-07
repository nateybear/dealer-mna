using StatsBase: mean, weights as w, median
using Distributions: Normal, pdf
using Dates: Day
using Lazy: @>
using DataFrames
using PartialFunctions

const STATISTICS = (
    mean_baseprice=:mean_price => mean,
    sales_baseprice=:mean_price => sum,
    median_markup=:price_normed => median,
    total_sales=:vehsalesprice => sum,
    total_units=:vehsalesprice => length,
)

const BOOTSTRAP_REPLICAS = 250

month_counter(d1) = d2 -> floor(Int, (d2 - d1) / Day(30))
create_time_col(df, date) = @> df begin
    transform(:regeffdate => ByRow(month_counter(date)) => :T)
    subset(:T => ByRow(in(-6:5)))
end

partial_out(df, outcome_col, by_col) = @> df groupby(by_col) transform(outcome_col => (x -> x .- mean(x)) => outcome_col)

function event_study(
    treated_before,
    treated_after,
    control_before,
    control_after,
    dealer_demographics,
)

    distance = make_distance_fn(dealer_demographics)
    treated_uid = only(unique(treated_before.dealer_uid))
    treated_demographics = only(eachrow(subset(dealer_demographics, :uid => ByRow(==(treated_uid)))))
    control_demographics = semijoin(dealer_demographics, control_before, on=:uid => :dealer_uid)
    control_distances = [distance(treated_demographics, d) for d in eachrow(control_demographics)]

    control_weights = DataFrame(
        dealer_uid=control_demographics.uid,
        weight=pdf.(Normal(), control_distances)
    )

    last_seller_regeffdate = maximum(treated_before.regeffdate)
    first_buyer_regeffdate = minimum(treated_after.regeffdate)

    treated_before = create_time_col(treated_before, last_seller_regeffdate + Day(1))
    treated_after = create_time_col(treated_after, first_buyer_regeffdate - Day(1))
    control_before = create_time_col(control_before, last_seller_regeffdate + Day(1))
    control_after = create_time_col(control_after, first_buyer_regeffdate - Day(1))

    all_registrations = vcat(
        treated_before,
        treated_after,
        control_before,
        control_after
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
        control_outcomes = innerjoin(control_outcomes, DataFrame(dealer_uid=uids_resample), on=:dealer_uid)
        vcat(
            treated_outcome,
            control_outcomes
        )
    end

    function te_estimate(panel, (outcol, _))
        panel = @> panel begin
            partial_out(outcol, :T)
            partial_out(outcol, :dealer_uid)
        end
        combine(groupby(panel, :T)) do gdf
            if !any(gdf.dealer_uid .== treated_uid)
                return (; outcol => missing)
            end
            treated_outcome = only(gdf[gdf.dealer_uid.==treated_uid, outcol])
            control_outcomes = @> gdf subset(:dealer_uid => ByRow(!=(treated_uid))) innerjoin(control_weights, on=:dealer_uid)
            synthetic_control_outcome = mean(control_outcomes[!, outcol], w(control_outcomes.weight))

            (; outcol => (treated_outcome - synthetic_control_outcome))
        end
    end

    sample_estimate = mapreduce(innerjoin$(on=:T,), pairs(STATISTICS)) do s
        panel = make_panel(s)
        te_estimate(panel, s)
    end

    # B is the bootstrap number (missing means in-sample estimate)
    sample_estimate.B .= missing

    out = sample_estimate

    for i in Base.OneTo(BOOTSTRAP_REPLICAS)
        @info "Bootstrap iteration $(i)"
        estimate = mapreduce(innerjoin$(on=:T,), pairs(STATISTICS)) do s
            panel = make_panel(s)
            te_estimate(shuffle(panel), s)
        end
        estimate.B .= i
        out = vcat(out, estimate)
    end

    return out
end
