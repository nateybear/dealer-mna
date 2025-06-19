using DealerMnA
using UUIDs
using Plots

!isempty(ARGS) && snakemake!(ARGS[1])

with_logging() do
    Dealers.load_from_disk!(snakemake.input["dealers_init"])
    transaction_uids = read_csv(snakemake.input["transaction_uids"])
    transactions_for_analysis = JLD2.load_object(
        snakemake.input["transactions_for_analysis"]
    )

    out = DataFrame(
        change_market_share = Float64[],
        transaction_uid = UUID[],
        dealer_uid = UUID[],
        group = String[],
    )

    for (key, transacted_group) in pairs(groupby(transaction_uids, :transaction_uid))
        transaction_uid = UUID(key.transaction_uid)
        transacted_dealers = UUID.(transacted_group.dealer_uid)

        t = first(
            t for t in transactions_for_analysis
            if t.uid == transaction_uid
        )

        period = let d = t.last_seller_regeffdate
            (d - Quarter(3)):(d - Quarter(2))
        end

        buyer_seller_dealers = Dict(
            "seller" =>
                eachline("data/affiliated_dealerships/seller_group_$(transaction_uid).txt"),
            "buyer" =>
                eachline("data/affiliated_dealerships/buyer_group_$(transaction_uid).txt"),
        )
        for (group, dealers) in buyer_seller_dealers
            for uid in dealers
                dealer_uid = UUID(uid) # uid of *affiliated* dealer, not the transacted dealership
                @info "Evaluating $(group) market share" transaction_uid dealer_uid

                registrations = DataFrame(
                    Arrow.Table(
                        "data/treated/$(group)/last_registrations_before_$(transaction_uid)_$(dealer_uid).arrow"
                    ),
                )

                change_market_share = Empirics.market_share(
                    registrations, period, Dict(transacted_dealers .=> t.buyer_gid)
                )

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
