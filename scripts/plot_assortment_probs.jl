using DealerMnA
using Plots
using StatsBase: mean, quantile
using Printf
using DataFrames
using FixedEffectModels
using RegressionTables

import FixedEffectModels: @formula as @f

# Three quantities of interest (dealer-quarter aggregates). After residualizing
# against demographics and subtracting within-dealer means, the high-ε tercile
# should show a price ↓ → quantity ↑ → trim-count ↑ dynamic; the low/mid
# terciles should not.
const OUTCOMES = [:total_units, :unique_trims, :median_markup]

lag(x) = [missing; collect(x)[1:(end - 1)]]

# Per dealer, lag by one row (data already sorted by t within dealer).
function lagged_within_dealer!(dq, cols = [:total_units, :median_markup])
    sort!(dq, [:dealer_uid, :t])
    transform!(
        groupby(dq, :dealer_uid),
        cols .=> lag,
    )
    return nothing
end

function trim_variation_table(fords)
    trim_counts = @> fords groupby([:model, :extra]) combine(nrow => :total_count)

    dealer_quarter_trim = @> fords groupby([:model, :extra, :dealer_uid, :t]) combine(
        nrow => :_sales
    )
    dq_with_trim = @> dealer_quarter_trim begin
        groupby([:model, :extra])
        combine(nrow => :n_dealer_quarters_with_trim)
    end
    n_total_dq = nrow(unique(select(fords, :dealer_uid, :t)))

    leftjoin!(trim_counts, dq_with_trim, on = [:model, :extra])
    trim_counts.n_dealer_quarters_total .= n_total_dq
    trim_counts.share_dealer_quarters_with_trim =
        trim_counts.n_dealer_quarters_with_trim ./ n_total_dq
    sort!(trim_counts, :total_count, rev = true)
    return trim_counts
end

function (@main)(_)
    snakemake!()
    with_logging() do
        fords = read_arrow(snakemake.input["fords"])
        fords = subset(fords, :t => ByRow(<(Date(2020, 1, 1))))

        @info "Aggregating dealer-quarter panel"
        dq = Supply.make_dealer_quarter(fords)

        @info "Residualizing by demographics within quarter" nrows = nrow(dq)

        # ε = total_units residual (pre-demean). Drives the tercile cut.
        dealer_avg = @> dq groupby(:dealer_uid) combine(:total_units => mean => :avg_sales)
        cuts = quantile(dealer_avg.avg_sales, [1 / 3, 2 / 3])
        dealer_avg.tercile = map(dealer_avg.avg_sales) do ε
            if ε < cuts[1]
                "Low"
            elseif ε < cuts[2]
                "Mid"
            else
                "High"
            end
        end

        leftjoin!(dq, dealer_avg, on = :dealer_uid)
        lagged_within_dealer!(dq)

        dq.is_high = dq.tercile .== "High"
        dq.is_mid = dq.tercile .== "Mid"

        model(form) = reg(dq, form.lhs ~ form.rhs + fe(:t) + fe(:dealer_uid), Vcov.robust())

        formulae = [
            @f(log(total_units) ~ median_markup),
            @f(log(total_units) ~ median_markup & (1 + is_mid + is_high)),
            @f(unique_trims ~ log(total_units_lag)),
            @f(unique_trims ~ log(total_units_lag) & (1 + is_mid + is_high)),
            @f(unique_trims ~ median_markup_lag),
            @f(unique_trims ~ median_markup_lag & (1 + is_mid + is_high))
        ]

        t = regtable(model.(formulae)..., render = LatexTable(), regression_statistics = [Nobs])
        open(snakemake.output["reg_table"], "w") do io
            print(io, t)
        end

        @info "Writing trim-variation table"
        trim_table = trim_variation_table(fords)
        CSV.write(snakemake.output["trim_csv"], trim_table)
        @info "  rows" nrow(trim_table) top_share = first(
            trim_table.share_dealer_quarters_with_trim, 5
        )
    end
    return nothing
end
