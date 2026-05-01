using DealerMnA

snakemake!()

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
    changes = Iterators.flatmap(groupby(transactions, :transaction_uid)) do transaction
        map(eachrow(transaction)) do row
            ms = Empirics.market_share(
                row.dealer_uid,
                Dealers.eventtime.(Ref(row.t), -4:-3),
                Dict(transaction.dealer_uid .=> row.t.buyer_gid),
            )
            hhi = Empirics.hhi(
                row.dealer_uid,
                Dealers.eventtime.(Ref(row.t), -4:-3),
                Dict(transaction.dealer_uid .=> row.t.buyer_gid),
            )
            return (
                transaction_uid = row.transaction_uid,
                dealer_uid = row.dealer_uid,
                change_market_share = ms[1],
                old_market_share = ms[2],
                new_market_share = ms[3],
                sales_volume = ms[4],
                change_hhi = hhi[1],
                old_hhi = hhi[2],
                new_hhi = hhi[3],
            )
        end
    end
    leftjoin!(transactions, DataFrame(changes), on = [:transaction_uid, :dealer_uid])
    select!(
        transactions,
        [
            :transaction_uid,
            :dealer_uid,
            :change_market_share,
            :old_market_share,
            :new_market_share,
            :sales_volume,
            :change_hhi,
            :old_hhi,
            :new_hhi,
        ],
    )
    Arrow.write(
        snakemake.output[1],
        transactions,
    )
end
