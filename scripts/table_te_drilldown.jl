using DealerMnA

using StatsBase
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

function print_table(filename, df)
    lowerprice = subset(df, :p_comparison => ByRow(contains("<")))
    sameprice = subset(df, :p_comparison => ByRow(contains("=")))
    higherprice = subset(df, :p_comparison => ByRow(contains(">")))
    open(filename, "w") do io
        latex_tabular(
            io,
            Tabular("c"^4),
            [Rule(:top),
                [
                    "Effect",
                    "Avg. Trim Rank",
                    "Num. Available Trims",
                    "Avg. Model Base Price",
                    "Avg. Markup",
                ],
                Rule(:mid),
                ["P < 0"],
                Rule(:mid),
                collect.(eachrow(make_count_table(lowerprice)))...,
                Rule(:mid),
                ["P = 0"],
                Rule(:mid),
                collect.(eachrow(make_count_table(sameprice)))...,
                Rule(:mid),
                ["P > 0"],
                Rule(:mid),
                collect.(eachrow(make_count_table(higherprice)))...,
                Rule(:bottom),
            ])
    end
    return nothing
end
function make_count_table(df)
    out = mapreduce(outerjoin$(on = :order,), (:trim_rank, :num_trims, :baseprice, :markup)) do col
        counts = @> df groupby(col) combine(nrow) sort(col)
        counts[!, col] .= last.(counts[!, col], 3)
        return select(counts, col => :order, :nrow => col)
    end
    sort!(out, :order)
    transform!(out, names(out) .=> ByRow(x -> coalesce(x, 0)), renamecols = false)
    return out
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
    transactions.trim_rank = comp_vector(:trim_rank, "Avg. Trim")
    transactions.num_trims = comp_vector(:unique_trims, "Num. Available Trims")
    transactions.baseprice = comp_vector(:mean_baseprice, "Model Base Price")
    transactions.markup = comp_vector(:median_markup, "Avg. Markup")

    goodies = subset(transactions, :q_comparison => ByRow(contains(">")))
    print_table(snakemake.output["q_increased"], goodies)
    baddies = subset(transactions, :q_comparison => ByRow(contains("<")))
    print_table(snakemake.output["q_decreased"], baddies)

    # buyer and seller groups
    transform!(
        transactions,
        :transaction_uid =>
            ByRow(
                x -> Set(eachline("data/affiliated_dealerships/seller_group_$(x).txt"))
            ) => :seller_group,
    )
    transform!(
        transactions,
        :transaction_uid =>
            ByRow(x -> Set(eachline("data/affiliated_dealerships/buyer_group_$(x).txt"))) =>
                :buyer_group,
    )
    avg_length(x) = mean(length.(x))

    table = @> transactions begin
        groupby([:p_comparison, :q_comparison])
        combine(
            :seller_group => avg_length,
            :buyer_group =>
                avg_length,
        )
        transform(
            [:seller_group_avg_length, :buyer_group_avg_length] =>
                ByRow((x, y) -> "$(round(x, digits=3)) -> $(round(y, digits=3))") => :out,
        )
        sort([:p_comparison, :q_comparison])
        select(r"comparison", :out)
        unstack(:p_comparison, :q_comparison, :out)
    end

    open(snakemake.output["group_sizes"], "w") do io
        latex_tabular(
            io,
            Tabular("l|" * "c"^length(names(table, Not(:p_comparison))) ),
            [
                Rule(:top),
                ["", names(table, Not(:p_comparison))...],
                Rule(:mid),
                collect.(eachrow(table))...,
                Rule(:bottom),
            ],
        )
    end
end
