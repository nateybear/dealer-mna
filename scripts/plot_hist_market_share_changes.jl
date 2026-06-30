using DealerMnA
using Plots

snakemake!()

with_logging() do
    transactions = read_arrow(snakemake.input["transactions_market_share_change"])

    histogram(
        transactions.change_market_share ./ transactions.old_market_share,
        xlabel = "Percent Change in Market Share",
        label = nothing,
        bins = 20,
    )
    png(snakemake.output["histogram"])
    p = plot(
        xticks = ((1, 2), ("Before", "After")),
        yaxis = :log10,
        ylabel = "Market Share (log scale)",
    )
    sales_range = extrema(transactions.sales_volume)
    pointsize_range = (2, 12)
    for t in eachrow(transactions)
        color = if t.change_market_share > 0
            :blue
        elseif t.change_market_share < 0
            :orange
        else
            :gray
        end
        plot!(
            p,
            [1, 2],
            [t.old_market_share, t.new_market_share], # add small constant to avoid log(0)
            label = nothing,
            color = color,
            alpha = 0.5,
            yaxis = :log10,
            )
        pointsize =
            ((t.sales_volume - sales_range[1]) / (sales_range[2] - sales_range[1])) *
            (pointsize_range[2] - pointsize_range[1]) + pointsize_range[1]
        scatter!(
            p,
            [1, 2],
            [t.old_market_share, t.new_market_share], # add small constant to avoid log(0)
            label = nothing,
            color = color,
            alpha = 0.5,
            markersize = pointsize,
            yaxis = :log10,
            )
    end
    png(snakemake.output["before_after_market_share"])
    p = plot(xticks = ((1, 2), ("Before", "After")))
    for t in eachrow(transactions)
        color = if t.change_hhi > 0
            :blue
        elseif t.change_hhi < 0
            :orange
        else
            :gray
        end
        plot!(
            p,
            [1, 2],
            [t.old_hhi, t.new_hhi],
            label = nothing,
            color = color,
            alpha = 0.5,
            yaxis = :log10,
        )
        pointsize = ((t.sales_volume - sales_range[1]) / (sales_range[2] - sales_range[1])) *
            (pointsize_range[2] - pointsize_range[1]) + pointsize_range[1]
        scatter!(
            p,
            [1, 2],
            [t.old_hhi, t.new_hhi],
            label = nothing,
            color = color,
            alpha = 0.5,
            markersize = pointsize,
            yaxis = :log10
        )
    end
    png(snakemake.output["before_after_hhi"])
    return nothing
end
