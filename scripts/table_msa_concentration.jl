using DealerMnA
using Base.Iterators: flatten
using DataStructures: counter
using LaTeXTabulars

!isempty(ARGS) && snakemake!(ARGS[1])

function cr4(gids)
    counts = gids |> counter |> values |> collect |> sort$(rev = true,) |> cumsum
    return round(length(counts) > 4 ? counts[4] / counts[end] : 1.0, digits = 3)
end

function count_overlap(transaction_uid_counter, uids)
    return sum(get(transaction_uid_counter, uid, 0) for uid in uids)
end

with_logging() do
    Dealers.load_from_disk!(snakemake.input["dealers_init"])
    transactions = JLD2.load_object(snakemake.input["transactions_for_analysis"])
    transaction_indices = collect(flatten(Dealers.dealer_idxs(t) for t in transactions))
    transaction_uid_counter = counter(Dealers.State.df.uid[transaction_indices])
    out = @> Dealers.State.df begin
        transform(:msa => ByRow(x -> something(x, "OUTSIDE MSA")) => :msa)
        groupby(:msa)
        combine(
            nrow => :num_dealerships,
            :uid => count_overlap$(transaction_uid_counter) => :num_transactions,
            :gid => cr4 => :cr4,
        )
        sort(:num_dealerships, rev = true)
    end
    open(snakemake.output[1], "w") do io
        latex_tabular(
            io,
            Tabular("lrrr"),
            [
                Rule(:top),
                ["MSA", "Dealerships", "Transactions", "CR4 (Num. Dealerships)"],
                Rule(:mid),
                collect.(eachrow(out))...,
                Rule(:bottom),
            ],
        )
    end
end