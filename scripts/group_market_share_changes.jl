using DealerMnA
using UUIDs
using Plots

!isempty(ARGS) && snakemake!(ARGS[1])

with_logging() do
    Dealers.load_from_disk!(snakemake.input["dealers_init"])
    transaction_uids = read_csv(snakemake.input["transaction_uids"])
    mapcols!(ByRow(UUID), transaction_uids)
    transactions_for_analysis = JLD2.load_object(
        snakemake.input["transactions_for_analysis"]
    )

    out = DataFrame(
        change_market_share = Float64[],
        transaction_uid = UUID[],
        dealer_uid = UUID[],
        group = String[],
    )

    count = 1

    all_groups = pairs(groupby(transaction_uids, :transaction_uid))
    for (key, transacted_group) in all_groups
        transaction_uid = key.transaction_uid
        transacted_dealers = transacted_group.dealer_uid

        t = first(
            t for t in transactions_for_analysis
            if t.uid == transaction_uid
        )

        period = [Dealers.eventtime(t, -3)]

        buyer_seller_dealers = Dict(
            "seller" =>
                eachline("data/affiliated_dealerships/seller_group_$(transaction_uid).txt"),
            "buyer" =>
                eachline("data/affiliated_dealerships/buyer_group_$(transaction_uid).txt"),
        )
        for (group, dealers) in buyer_seller_dealers
            for uid in dealers
                dealer_uid = UUID(uid) # uid of *affiliated* dealer, not the transacted dealership
                @info "$count Evaluating $(group) market share" transaction_uid dealer_uid
                count += 1

                change_market_share = Empirics.market_share(
                    dealer_uid,
                    period,
                    Dict(transacted_dealers .=> t.buyer_gid)
                )

                @info "  Change in market share: $(change_market_share)"

                # change in market share is always positive,
                # but for sellers it should be < 0
                if group == "seller"
                    change_market_share *= -1
                end

                push!(
                    out,
                    (;
                        change_market_share,
                        transaction_uid,
                        dealer_uid,
                        group,
                    ),
                )
            end
        end
    end

    transform!(out, names(out, r"uid") .=> ByRow(string), renamecols = false)
    Arrow.write(snakemake.output[1], out)
end
