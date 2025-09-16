using DealerMnA
using StatsBase: mean

using LinearAlgebra
using LaTeXTabulars
using LaTeXStrings

!isempty(ARGS) && snakemake!(ARGS[1])

function fit_plus_std(x, y; digits = 3)
    let significant = abs(x) > 1.96y, x = round(x; digits), y = round(y; digits)
        LaTeXString("\\makecell{$x$(significant ? "^{***}" : "") \\\\ ($y)\\\\[0.25em]}")
    end
end

with_logging() do
    df = DataFrame(Arrow.Table(snakemake.input["treatment_effects"]))
    extras = JLD2.load_object(snakemake.input["extras"])
    cols = [:mean_price, :total_units, :trim_rank, :mean_baseprice, :median_markup]
    table = DataFrame(
        map(cols) do col
            extra = only(eachrow(subset(extras, :stat => ByRow(==(col)))))
            offset = mean(extra.fes.fe_treated - extra.fes.fe_synthetic)
            before = @> df begin
                subset(:T => ByRow(==(-3)))
                select(col => "estimate", "$(col)_se" => "se")
                eachrow()
                only()
            end
            after = @> df begin
                subset(:T => ByRow(==(2)))
                select(col => "estimate", "$(col)_se" => "se")
                eachrow()
                only()
            end
            difference = after.estimate - before.estimate
            ∇ = [0, -1, 0, 0, 0, 1, 0]
            se_difference = sqrt(∇' * extra.V * ∇)
            before.estimate += offset
            after.estimate += offset
            before = fit_plus_std(before.estimate, before.se)
            after = fit_plus_std(after.estimate, after.se)
            difference = fit_plus_std(difference, se_difference)
            return (; col, before, after, difference)
        end,
    )

    titleize(col) = replace(string(col), "_" => " ") |> titlecase
    table.col = titleize.(table.col)

    open(snakemake.output[1], "w") do io
        latex_tabular(
            io,
            Tabular("l|" * "c"^3),
            [
                Rule(:top),
                ["", "Before", "After", "Difference"],
                Rule(:mid),
                collect.(eachrow(table))...,
                Rule(:mid),
                # give number of observations (thanks GPT)
                [
                    "Number of observations",
                    string(nrow(df[df.T .== -3, :])),
                    string(nrow(df[df.T .== 2, :])),
                    "",
                ],
                Rule(:bottom),
            ],
        )
    end
end
