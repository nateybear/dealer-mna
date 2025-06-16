using DealerMnA

using LinearAlgebra
using LaTeXTabulars
using LaTeXStrings

!isempty(ARGS) && snakemake!(ARGS[1])

function compare(column, outname = column; transaction_uid, dealer_uid)
    let
        estimate, se = Empirics.treatment_effect(column; transaction_uid, dealer_uid)
        if estimate > 0
            estimate - 1.96se > 0 ? "$(outname) > 0" :
            "$(outname) = 0"
        else
            estimate + 1.96se < 0 ? "$(outname) < 0" :
            "$(outname) = 0"
        end
    end
end

with_logging() do
    transactions = read_csv(snakemake.input["transaction_uids"])
    transactions.p_comparison = map(eachrow(transactions)) do col
        tes = DataFrame(
            Arrow.Table(
                "data/treatment_effects/treated_dealers/$(col.transaction_uid)/$(col.dealer_uid).arrow"
            ),
        )
        extras = JLD2.load_object(
            "data/treatment_effects/treated_dealers/$(col.transaction_uid)/$(col.dealer_uid)_extras.jld2"
        )
        compare(tes, extras, :mean_price, "P")
    end
    transactions.q_comparison = map(eachrow(transactions)) do col
        tes = DataFrame(
            Arrow.Table(
                "data/treatment_effects/treated_dealers/$(col.transaction_uid)/$(col.dealer_uid).arrow"
            ),
        )
        extras = JLD2.load_object(
            "data/treatment_effects/treated_dealers/$(col.transaction_uid)/$(col.dealer_uid)_extras.jld2"
        )
        compare(tes, extras, :total_units, "Q")
    end

    table = @> transactions begin
        groupby([:p_comparison, :q_comparison])
        combine(nrow)
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
