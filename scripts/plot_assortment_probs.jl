using DealerMnA
using Plots
using StatsBase: mean, quantile
using Printf
using DataFrames

# Three quantities of interest (dealer-quarter aggregates). After residualizing
# against demographics and subtracting within-dealer means, the high-ε tercile
# should show a price ↓ → quantity ↑ → trim-count ↑ dynamic; the low/mid
# terciles should not.
const OUTCOMES = [:total_units, :unique_trims, :median_markup]

function residualize_within_quarter!(dq, demos)
    transform!(groupby(dq, :t)) do df
        X = Matrix(df[!, demos])
        yu = Float32.(df.total_units)
        yt = Float32.(df.unique_trims)
        yp = Float32.(df.median_markup)
        return (;
            total_units_resid = yu .- Supply.npregress(X, yu)[:, 1],
            unique_trims_resid = yt .- Supply.npregress(X, yt)[:, 1],
            median_markup_resid = yp .- Supply.npregress(X, yp)[:, 1],
        )
    end
    return nothing
end

function add_within_dealer_demean!(dq)
    for col in OUTCOMES
        in_col = Symbol(col, :_resid)
        out_col = Symbol(col, :_dr)
        transform!(
            groupby(dq, :dealer_uid),
            in_col => (x -> x .- mean(x)) => out_col,
        )
    end
    return nothing
end

# Per dealer, lag by one row (data already sorted by t within dealer).
function lagged_within_dealer!(dq)
    sort!(dq, [:dealer_uid, :t])
    transform!(
        groupby(dq, :dealer_uid),
        :median_markup_dr =>
            (x -> [missing; collect(x)[1:(end - 1)]]) => :lag_median_markup_dr,
        :total_units_dr =>
            (x -> [missing; collect(x)[1:(end - 1)]]) => :lag_total_units_dr,
    )
    return nothing
end

# Equal-width quantile bins; aggregate (x_mean, y_mean) per bin.
function binned_means(x, y; nbins = 15)
    edges = quantile(x, range(0.0, 1.0, length = nbins + 1))
    bin = clamp.(searchsortedlast.(Ref(edges), x), 1, nbins)
    df = DataFrame(; bin, x, y)
    out = combine(groupby(df, :bin), :x => mean => :x_mean, :y => mean => :y_mean)
    sort!(out, :bin)
    return out
end

function tercile_panel(dq, x_col, y_col; xlabel, ylabel)
    plots = map(["Low", "Mid", "High"]) do t
        sub = subset(dq, :tercile => ByRow(==(t)))
        bm = binned_means(sub[!, x_col], sub[!, y_col])
        p = scatter(
            bm.x_mean,
            bm.y_mean;
            xlabel = xlabel,
            ylabel = ylabel,
            title = "$t ε tercile",
            legend = false,
            markersize = 4,
            markerstrokewidth = 0,
        )
        # OLS line on the binned means for visual reference.
        if nrow(bm) >= 2
            β = [ones(nrow(bm)) bm.x_mean] \ bm.y_mean
            xs = [minimum(bm.x_mean), maximum(bm.x_mean)]
            plot!(p, xs, β[1] .+ β[2] .* xs, lw = 2)
        end
        return p
    end
    return plot(plots...; layout = (1, 3), size = (1500, 400))
end

function trim_variation_table(fords)
    trim_counts = @> fords groupby([:model, :extra]) combine(nrow => :total_count)

    dealer_quarter_trim =
        @> fords groupby([:model, :extra, :dealer_uid, :t]) combine(nrow => :_sales)
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
        dealer_demographics = read_arrow(snakemake.input["dealer_demographics"])
        demos = Empirics.DEFAULT_DEMOGRAPHICS

        @info "Aggregating dealer-quarter panel"
        dq = Supply.make_dealer_quarter(fords)
        leftjoin!(
            dq, select(dealer_demographics, :uid, demos...), on = :dealer_uid => :uid
        )
        dropmissing!(dq, demos)

        @info "Residualizing by demographics within quarter" nrows = nrow(dq)
        residualize_within_quarter!(dq, demos)
        add_within_dealer_demean!(dq)

        # ε = total_units residual (pre-demean). Drives the tercile cut.
        dq.ε = dq.total_units_resid
        dealer_avg = @> dq groupby(:dealer_uid) combine(:ε => mean => :avg_ε)
        cuts = quantile(dealer_avg.avg_ε, [1 / 3, 2 / 3])
        dealer_avg.tercile = map(dealer_avg.avg_ε) do ε
            if ε < cuts[1]
                "Low"
            elseif ε < cuts[2]
                "Mid"
            else
                "High"
            end
        end
        leftjoin!(dq, select(dealer_avg, :dealer_uid, :tercile), on = :dealer_uid)

        lagged_within_dealer!(dq)
        dropmissing!(dq, [:lag_median_markup_dr, :lag_total_units_dr])

        outdir = snakemake.output["plots"]
        isdir(outdir) || mkdir(outdir)

        @info "Plotting price → quantity dynamic (per tercile)"
        p1 = tercile_panel(
            dq,
            :lag_median_markup_dr,
            :total_units_resid;
            xlabel = "Lagged price residual (within-dealer demeaned)",
            ylabel = "Quantity residual",
        )
        png(p1, joinpath(outdir, "price_to_quantity.png"))

        @info "Plotting quantity → trim dynamic (per tercile)"
        p2 = tercile_panel(
            dq,
            :lag_total_units_dr,
            :unique_trims_resid;
            xlabel = "Lagged quantity residual (within-dealer demeaned)",
            ylabel = "Trim-count residual",
        )
        png(p2, joinpath(outdir, "quantity_to_trims.png"))

        @info "Writing trim-variation table"
        trim_table = trim_variation_table(fords)
        CSV.write(snakemake.output["trim_csv"], trim_table)
        @info "  rows" nrow(trim_table) top_share = first(trim_table.share_dealer_quarters_with_trim, 5)
    end
    return nothing
end
