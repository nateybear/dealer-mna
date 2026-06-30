#### Dealer × quarter × product panel construction.
####
#### Lifted from scripts/plot_assortment_probs.jl so the panel is shared between
#### compliance estimation, sales-target estimation, and the assortment-probs
#### diagnostics.

using DataFrames
using Dates
using Lazy: @>
using OhMyThreads: tmapreduce
import ..Empirics

selectunique(X, cols) = select(unique(X, cols), cols)

extract_stat((v, (k, col))) = k => col => v
extract_stats(stats) = extract_stat.(collect(pairs(stats)))

lag(x) =
    let N = length(x)
        vcat(missing, first(x, N - 1))
    end

trim_cut(x) =
    if x < 1 / 3
        "Low"
    elseif x < 2 / 3
        "Mid"
    else
        "High"
    end

"""
    make_dealer_quarter(cars; statistics = Empirics.STATISTICS)

Aggregate per-car registrations to one row per (`t`, `dealer_uid`) with the
statistics defined in `statistics` (default: `Empirics.STATISTICS`).
"""
function make_dealer_quarter(cars::AbstractDataFrame; statistics = Empirics.STATISTICS)
    return @> cars groupby([:t, :dealer_uid]) combine(extract_stats(statistics)...)
end

"""
    make_panel(cars, dealers; statistics, demographics, extra_dealer_cols)

Build a dealer × quarter × product panel.

* `cars` — per-vehicle registrations (must have `t, dealer_uid, model, model_year, extra` and the columns referenced by `statistics`).
* `dealers` — dealer-level table with `uid` and any columns named in `demographics` ∪ `extra_dealer_cols`.

Returns a DataFrame keyed on `(t, dealer_uid, model, model_year, extra)` with
`q` (count), `is_sold` (binary), lagged dealer-quarter statistics named
`lag_<stat>`, and the dealer-level demographic columns.
"""
function make_panel(
    cars::AbstractDataFrame,
    dealers::AbstractDataFrame;
    statistics = Empirics.STATISTICS,
    demographics = Empirics.DEFAULT_DEMOGRAPHICS,
    extra_dealer_cols::Vector{Symbol} = Symbol[],
)
    D = vcat(demographics, extra_dealer_cols)

    out = @> cars begin
        groupby([:t, :dealer_uid, :model, :model_year, :extra])
        combine(nrow => :q)
    end

    let appropriate_t = (m, t) -> (Date(m) - Quarter(1)) <= t <= (Date(m) + Year(1))
        subset!(out, [:model_year, :t] => ByRow(appropriate_t))
    end

    agg_statistics = make_dealer_quarter(cars; statistics = statistics)
    transform!(agg_statistics, :t => ByRow(Base.Fix2(-, Quarter(1))) => :t)
    DataFrames.rename!(Base.Fix1(*, "lag_"), agg_statistics, cols = Not([:t, :dealer_uid]))
    out = innerjoin(out, agg_statistics, on = [:t, :dealer_uid])

    out = tmapreduce(vcat, groupby(out, :t)) do sales
        J = @> sales selectunique([:model, :model_year, :extra])
        J.is_new_model_year = J.model_year .== maximum(J.model_year)
        F = @> sales selectunique(:dealer_uid)
        JF = crossjoin(J, F)
        JF.t .= first(sales.t)
        leftjoin!(
            JF,
            select(unique(sales, :dealer_uid), :dealer_uid, names(sales, r"lag")),
            on = :dealer_uid,
        )
        leftjoin!(
            JF,
            select(sales, Not(names(sales, r"lag"))),
            on = [:t, :dealer_uid, :model, :model_year, :extra],
        )
        JF.q = coalesce.(JF.q, 0)
        JF.is_sold = JF.q .> 0
        leftjoin!(JF, select(dealers, :uid, D), on = :dealer_uid => :uid)
        return JF
    end
    subset!(groupby(out, [:model, :model_year, :extra]), :t => (>(1) ∘ length ∘ unique))
    return out
end
