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
        treatment_effect = Float64[],
        std_error = Float64[],
        transaction_uid = UUID[],
        dealer_uid = UUID[],
        group = String[],
    )

    for r in eachrow(transaction_uids)
        transaction_uid = UUID(r.transaction_uid)
        transacted_dealer_uid = UUID(r.dealer_uid)

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
                treatment_effect = DataFrame(
                    Arrow.Table(
                        "data/treatment_effects/$(group)/$(transaction_uid)/$(dealer_uid).arrow"
                    ),
                )
                extras = JLD2.load_object(
                    "data/treatment_effects/$(group)/$(transaction_uid)/$(dealer_uid)_extras.jld2"
                )
                treatment_effect, std_error = Empirics.treatment_effect(
                    treatment_effect,
                    extras,
                    :mean_price,
                )
                registrations = DataFrame(
                    Arrow.Table(
                        "data/treated/$(group)/last_registrations_before_$(transaction_uid)_$(dealer_uid).arrow"
                    ),
                )

                change_market_share = Empirics.market_share(
                    registrations, period, Dict(dealer_uid => t.buyer_gid)
                )

                push!(
                    out,
                    (;
                        change_market_share,
                        treatment_effect,
                        std_error,
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
