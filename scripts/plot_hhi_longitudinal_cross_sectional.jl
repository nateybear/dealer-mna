using DealerMnA
using Plots
using StatsBase: quantile
using Folds

import .Registrations: registrations
import .Empirics: UniformWeight
import .Dealers: timetravel

snakemake!()

# pick a high volume that is closest to the overall
# proportion of body classes sold (since I am 
# weighting by body class in the market share calc)
function representative_dealer()
    overall_props = @> registrations() groupby(:body_class) combine(proprow)
    dealer_props = combine(groupby(registrations(), :dealer_uid)) do group
        dealer_props = @> group begin
            groupby(:body_class)
            combine(proprow => :p, nrow)
            rightjoin(overall_props, on = :body_class)
            transform(:p => ByRow(x -> coalesce(x, 0)) => :p)
        end
        return (
            dist = sum((dealer_props.p .- dealer_props.proprow) .^ 2),
            N = sum(skipmissing(dealer_props.nrow)),
        )
    end
    subset!(dealer_props, :N => ByRow(≥(quantile(dealer_props.N, 0.75))))
    sort!(dealer_props, :dist)
    return dealer_props.dealer_uid[1]
end

with_logging() do
    Dealers.load_from_disk!(snakemake.input["dealers_init"])

    @info "Calculating a representative_dealer"
    focal_dealer_uid = representative_dealer()
    dates = Date(2017):Quarter(1):Date(2020)

    @info "Calculating longitudinal HHI"
    hhis = map(zip(dates, dates[2:end])) do (lo, hi)
        Empirics.hhi(
            focal_dealer_uid,
            lo:hi,
            weight = UniformWeight(),
        )
    end

    plot(
        dates[1:(end - 1)],
        hhis,
        label = nothing,
        xlabel = "Date",
        ylabel = "HHI",
        grid = true,
        linewidth = 2,
        right_margin = 5Plots.mm,
        left_margin = Plots.mm,
        bottom_margin = Plots.mm,
        xticks = (
            Date(2017):Month(6):Date(2019, 7, 1),
            Date(2017):Month(6):Date(2019, 7, 1),
        ),
    )
    png(snakemake.output["longitudinal"])

    @info "Calculating cross-sectional HHI"
    # cross-sectional at 2018-Q1
    active = timetravel(Date(2018, 3, 31)) do
        Dealers.State.df.active
    end
    hhi_cs = Folds.map(
        Dealers.State.df.uid[active],
        ThreadedEx(simd = true),
    ) do dealer_uid
        Empirics.hhi(
            dealer_uid,
            Date(2018, 1, 1):Date(2018, 3, 31),
        )
    end

    histogram(
        hhi_cs,
        xlabel = "HHI",
        ylabel = "Number of Dealers",
        label = nothing,
        grid = true,
        right_margin = 5Plots.mm,
        left_margin = Plots.mm,
        bottom_margin = Plots.mm,
        linewidth = 2,
    )

    cuts = [4, 5, 10]
    for i in cuts
        vline!([10_000 / i], label = "$i effective firms", linestyle = :dash, linewidth = 2)
    end
    png(snakemake.output["cross_sectional"])
end
