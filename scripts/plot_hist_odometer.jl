using DealerMnA
using Plots
using Loess

!isempty(ARGS) && snakemake!(ARGS[1])

with_logging() do
    registrations = DataFrame(Arrow.Table(snakemake.input["registrations"]))
    select!(registrations, :vehodmtrreading => :odometer, :regeffdate, :model_year)
    registrations = subset(
        registrations,
        :odometer => ByRow(<(500)),
        view = true,
        skipmissing = true,
    )
    registrations = dropmissing(registrations)
    year_is_off(regeffdate, model_year) =
        abs(year(regeffdate) - model_year) > 1
    transform!(
        registrations,
        [:regeffdate, :model_year] => ByRow(year_is_off) => :year_is_off,
    )
    x_grid = range(extrema(registrations.odometer)..., length = 100)
    y_pred = let model = loess(registrations.odometer, registrations.year_is_off)
        predict.(Ref(model), x_grid)
    end

    histogram(
        registrations.odometer,
        label = nothing,
        bins = 100,
        linewidth = 3,
        xlabel = "Odometer Reading",
        ylabel = "Frequency",
    )

    plot!(
        twinx(),
        x_grid,
        y_pred,
        label = nothing,
        color = "red",
        ylabel = "|Registration - Model Year| > 1",
    )

    png(snakemake.output[1])
end
