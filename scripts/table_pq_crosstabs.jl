using DealerMnA
using StatsBase: mean

using LinearAlgebra
using LaTeXTabulars
using LaTeXStrings

!isempty(ARGS) && snakemake!(ARGS[1])

function compare(offset, extras, outname, before = false)
    let
        delta = offset .+ extras.fes.fe_treated - extras.fes.fe_synthetic
        ∇ = [1 / 2, 1 / 2, 0, 0, 0, 0, 0]
        se = sqrt(∇' * extras.V * ∇)
        map(delta) do d
            if d > 0
                d - 1.96se > 0 ? "$(outname) > 0" :
                "$(outname) = 0"
            else
                d + 1.96se < 0 ? "$(outname) < 0" :
                "$(outname) = 0"
            end
        end
    end
end

with_logging() do
    df = subset(
        DataFrame(Arrow.Table(snakemake.input["treatment_effects"])),
        :T => ByRow(≤(-3)),
    )
    extras = JLD2.load_object(snakemake.input["extras"])
    mean_price = only(eachrow(subset(extras, :stat => ByRow(==(:mean_price)))))
    total_units = only(eachrow(subset(extras, :stat => ByRow(==(:total_units)))))
    mean_price.fes.p_comparison = compare(mean(df.mean_price), mean_price, "P")
    total_units.fes.q_comparison = compare(mean(df.total_units), total_units, "Q")

    transactions = innerjoin(
        select(mean_price.fes, [:reg_uid, :p_comparison]),
        select(total_units.fes, [:reg_uid, :q_comparison]),
        on = :reg_uid,
    )

    table = @> transactions begin
        groupby([:p_comparison, :q_comparison])
        combine(nrow)
        fillcombinations([:p_comparison, :q_comparison], fill = 0)
        sort([:p_comparison, :q_comparison])
        unstack(:p_comparison, :q_comparison, :nrow)
    end

    # Add totals row and column
    totals_column = sum.(eachrow(table[:, Not(:p_comparison)]))
    totals_row = sum.(eachcol(table[:, Not(:p_comparison)]))

    table.Total = totals_column

    open(snakemake.output[1], "w") do io
        latex_tabular(
            io,
            Tabular("l|" * "c"^length(names(table, Not([:p_comparison, :Total]))) * "|c"),
            [
                Rule(:top),
                ["", names(table, Not(:p_comparison))...],
                Rule(:mid),
                collect.(eachrow(table))...,
                Rule(:mid),
                ["Total", totals_row..., sum(totals_row)],
            ],
        )
    end
end
