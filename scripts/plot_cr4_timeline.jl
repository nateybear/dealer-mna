using DealerMnA
using Base.Iterators: flatten
using DataStructures: counter
using Plots

!isempty(ARGS) && snakemake!(ARGS[1])

function cr4(gids)
    counts = gids |> counter |> values |> collect |> sort$(rev = true,) |> cumsum
    return length(counts) > 4 ? counts[4] / counts[end] : 1.0
end

with_logging() do
    Dealers.load_from_disk!(snakemake.input["dealers_init"])
    events = reverse([e for e in Dealers.Events if e.when < Date(2020, 1, 1)])
    dates = map(e -> e.when, events)
    cr4s = map(events) do e
        Dealers.timetravel(e) do
            cr4(Dealers.State.df.gid)
        end
    end
    plot(
        dates,
        cr4s,
        label = "CR4",
        xlabel = "Date",
        ylabel = "CR4",
        legend = :topright,
        grid = true,
        linewidth = 2,
        right_margin = 5Plots.mm,
        left_margin = Plots.mm,
        bottom_margin = Plots.mm,
        xticks = (
            Date(2017, 7, 1):Month(6):Date(2020, 1, 1),
            Date(2017, 7, 1):Month(6):Date(2020, 1, 1),
        ),
    )
    png(snakemake.output[1])
end
