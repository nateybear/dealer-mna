using DealerMnA
using StatsBase
using LaTeXStrings
using LaTeXTabulars

wmean(x, w) = mean(x, weights(w))
wstd(x, w) = std(x, weights(w))

!isempty(ARGS) && snakemake!(ARGS[1])

function format(row, digits = 2)
    # make, avg. price, num trims, std price within, std price across, std ratio
    fit_plus_std(x, y) =
        let x = round(x; digits), y = round(y; digits)
            LaTeXString("\\makecell{$x \\\\ ($y)\\\\[0.25em]}")
        end
    return [
        row.make,
        fit_plus_std(row.mean_price, row.std_mean_price),
        fit_plus_std(row.mean_num_trims, row.std_num_trims),
        fit_plus_std(row.mean_std_price_within, row.std_std_price_within),
        fit_plus_std(row.mean_std_price_across, row.std_std_price_across),
        round(row.std_price_ratio; digits),
    ]
end

with_logging() do
    make_stats = @> Registrations.price_table() begin
        groupby([:make, :model, :model_year])
        combine(
            nrow => :num_trims,
            [:mean_price, :n] => wstd => :std_price_across,
            :n => sum => :n,
        )
        groupby(:make)
        combine(
            [:num_trims, :n] => wmean => :mean_num_trims,
            [:num_trims, :n] => wstd => :std_num_trims,
            [:std_price_across, :n] => wmean => :mean_std_price_across,
            [:std_price_across, :n] => wstd => :std_std_price_across,
        )
    end

    within_price = @> snakemake.input["new_registrations"] begin
        Arrow.Table()
        DataFrame()
        select(:make, :model, :model_year, :vehsalesprice, :trim, :series)
        Registrations.normalize_price()
        groupby([:make, :model, :model_year])
        combine(
            [:vehsalesprice, :mean_price] => std ∘ (-) => :std_price_within,
            :vehsalesprice => mean => :mean_price,
            :vehsalesprice => std => :std_price,
            nrow,
        )
        groupby(:make)
        combine(
            [:std_price_within, :nrow] => wmean => :mean_std_price_within,
            [:std_price_within, :nrow] => wstd => :std_std_price_within,
            [:mean_price, :nrow] => wmean => :mean_price,
            [:mean_price, :nrow] => wstd => :std_mean_price,
        )
    end

    make_stats = innerjoin(make_stats, within_price, on = :make)
    transform!(
        make_stats,
        [:mean_std_price_within, :mean_std_price_across] => ByRow(/) => :std_price_ratio,
    )
    sort!(make_stats, :make)

    open(snakemake.output[1], "w") do io
        latex_tabular(
            io,
            Tabular("l" * "c"^5),
            [Rule(:top),
                [
                    "Make",
                    "Avg. Price",
                    "No. Trims",
                    "Std. Price Within",
                    "Std. Price Across",
                    "Across / Within",
                ],
                Rule(:mid),
                format.(eachrow(make_stats))...,
                Rule(:bottom)],
        )
    end
end
