using DealerMnA

!isempty(ARGS) && snakemake!(ARGS[1])

with_logging() do
    Dealers.load_from_disk!(snakemake.input["dealers_init"])
    transactions = JLD2.load_object(snakemake.input["transactions_for_analysis"])
    transactions = DataFrame(
        t = transactions,
        uid = [t.uid for t in transactions],
    )
    transactions = innerjoin(
        mapcols!(ByRow(UUID), read_csv(snakemake.input["transaction_uids"])),
        transactions,
        on = :transaction_uid => :uid,
    )
    transactions.change_market_share = map(eachrow(transactions)) do row
        Empirics.market_share(
            row.dealer_uid,
            Dealers.eventtime.(Ref(row.t), -4:-3),
            Dict(row.dealer_uid => row.t.buyer_gid),
        )
    end
    select!(
        transactions,
        [:transaction_uid, :dealer_uid, :change_market_share],
    )
    Arrow.write(
        snakemake.output[1],
        transactions,
    )
end
