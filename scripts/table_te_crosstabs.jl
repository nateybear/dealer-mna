using DealerMnA

using LinearAlgebra
using LaTeXTabulars
using LaTeXStrings
import Distributions: MvNormal, cdf

!isempty(ARGS) && snakemake!(ARGS[1])

function compare(column, outname = column; transaction_uid, dealer_uid)
    let
        estimate, D_placebo = Empirics.treatment_effect(column; transaction_uid, dealer_uid)
        quantile = cdf(D_placebo, estimate)
        if quantile < 0.05
            "$(outname) < 0"
        elseif quantile > 0.95
            "$(outname) > 0"
        else
            "$(outname) = 0"
        end
    end
end

with_logging() do
    transactions = read_csv(snakemake.input["transaction_uids"])
    function comp_vector(col, outname)
        map(eachrow(transactions)) do row
            compare(col, outname; row.transaction_uid, row.dealer_uid)
        end
    end
    transactions.p_comparison = comp_vector(:mean_price, "P")
    transactions.q_comparison = comp_vector(:total_units, "Q")

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
