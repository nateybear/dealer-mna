using DealerMnA

using UUIDs

snakemake!()

with_logging() do
    Dealers.load_from_disk!(snakemake.input["dealers_init"])
    transactions = JLD2.load_object(snakemake.input["transactions"])
    transaction_uids = read_csv(snakemake.input["transaction_uids"])
    mapcols!(ByRow(UUID), transaction_uids)
    uid = UUID(snakemake.wildcards["transaction_uid"])
    t = first(t for t in transactions if t.uid == uid)
    transacted_uids = transaction_uids.dealer_uid
    Dealers.timetravel(t, before = true) do
        buyer_uids =
            subset(
                Dealers.State.df,
                :gid => ByRow(==(t.buyer_gid)),
                :active => ByRow(identity),
                :uid => ByRow(!in(transacted_uids)),
            ).uid
        @info "Found $(length(buyer_uids)) buyers for transaction $(t.uid)"
        open(snakemake.output["buyer_group"], "w") do f
            for dealer_uid in buyer_uids
                last_registrations = Empirics.last_registrations_before(t; dealer_uid)
                first_registrations = Empirics.first_registrations_after(t; dealer_uid)
                if !ismissing(last_registrations) &&
                    !ismissing(first_registrations) &&
                    !isempty(last_registrations) &&
                    !isempty(first_registrations)
                    println(f, dealer_uid)
                end
            end
        end
    end
    Dealers.timetravel(t, before = false) do
        seller_uids =
            subset(
                Dealers.State.df,
                :gid => ByRow(==(t.seller_gid)),
                :active => ByRow(identity),
                :uid => ByRow(!in(transacted_uids)),
            ).uid
        @info "Found $(length(seller_uids)) sellers for transaction $(t.uid)"
        open(snakemake.output["seller_group"], "w") do f
            for dealer_uid in seller_uids
                last_registrations = Empirics.last_registrations_before(t; dealer_uid)
                first_registrations = Empirics.first_registrations_after(t; dealer_uid)
                if !ismissing(last_registrations) &&
                    !ismissing(first_registrations) &&
                    !isempty(last_registrations) &&
                    !isempty(first_registrations)
                    println(f, dealer_uid)
                end
            end
        end
    end
end
