using DealerMnA
using Plots

snakemake!()

with_logging() do
    Dealers.load_from_disk!(snakemake.input["dealers_init"])
    transactions = JLD2.load_object(snakemake.input["transactions_for_analysis"])
    group_sizes = DataFrame(
        map(transactions) do t
            Dealers.timetravel(t) do
                buyer_group = subset(
                    Dealers.State.df,
                    :gid => ByRow(==(t.buyer_gid)),
                    :active => ByRow(identity),
                )
                seller_group = subset(
                    Dealers.State.df,
                    :gid => ByRow(==(t.seller_gid)),
                    :active => ByRow(identity),
                )
                (
                    buyer_size = nrow(buyer_group),
                    seller_size = nrow(seller_group) - 1,
                )
            end
        end,
    )
    hline([0.0], label = nothing)
    vline!([0.0], label = nothing)
    scatter!(
        group_sizes.seller_size + rand(nrow(group_sizes)) .* (group_sizes.seller_size .> 0),
        group_sizes.buyer_size + rand(nrow(group_sizes)) .* (group_sizes.buyer_size .> 0),
        xlabel = "Seller Group Size",
        ylabel = "Buyer Group Size",
        label = nothing,
        markercolor = :algae,
        markersize = 5,
    )
    annotate!(
        [0.05, 10.0],
        [30.0, 2.0],
        [
            text(
                "← Exiting Sellers",
                8,
                :black,
                :top,
                :left,
            ),
            text(
                "↓ Entering Buyers",
                8,
                :black,
                :top,
                :left,
            ),
        ],
    )
    png(snakemake.output[1])
end
