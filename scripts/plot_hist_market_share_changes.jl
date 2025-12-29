using DealerMnA
using Plots

!isempty(ARGS) && snakemake!(ARGS[1])

with_logging() do
    transactions = read_arrow(snakemake.input["transactions_market_share_change"])
    histogram(
        filter(!iszero, transactions.change_market_share),
        xlabel = "Change in Market Share",
        label = nothing,
        bins = 20,
    )
    png(snakemake.output[1])
end
