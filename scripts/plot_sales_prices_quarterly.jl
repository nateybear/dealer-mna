using DealerMnA
using Plots
using StatsBase

snakemake!()

with_logging() do
    registrations = read_arrow(snakemake.input["new_registrations"])
    select!(registrations, :regeffdate, :vehsalesprice)
    transform!(
        registrations, :regeffdate .=> ByRow.([year, quarterofyear]) .=> [:year, :quarter]
    )
    registrations = @> registrations begin
        groupby([:year, :quarter])
        combine(nrow, :vehsalesprice => median => :avg_price)
        subset(:year => ByRow(<(2020)))
    end
    registrations.date = @. Date(registrations.year) + Quarter(registrations.quarter - 1)
    sort!(registrations, :date)

    plot(
        registrations.date,
        100registrations.nrow ./ first(registrations.nrow),
        label = "Quarterly Sales (2016Q1 = $(first(registrations.nrow)))",
        ylabel = "Index",
        xlabel = "Date",
        size = (800, 600),
        seriestype = :line,
        marker = :circle,
        color = "blue",
    )

    hline!([100], label = nothing, color = "black", linestyle = :dash)

    plot!(
        registrations.date,
        100registrations.avg_price ./ first(registrations.avg_price),
        label = "Average Price (2016Q1 = \$$(round(Int64, first(registrations.avg_price))))",
        seriestype = :line,
        marker = :circle,
        color = "red",
    )

    png(snakemake.output[1])
end
