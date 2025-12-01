using DealerMnA
using DrWatson
import JLD2
import Arrow
using UUIDs

!isempty(ARGS) && snakemake!(ARGS[1])

with_logging() do
    Dealers.load_from_disk!(snakemake.input["dealers_init"])
    transactions = JLD2.load_object(snakemake.input["transactions_for_analysis"])
    transactions = DataFrame(
        uid = [t.uid for t in transactions],
        t = transactions,
    )

    iter = if haskey(snakemake.params, "dealer_uids")
        dealer_uids = DataFrame(
            transaction_uid = collect(keys(snakemake.params["dealer_uids"])),
            dealer_uid = collect(values(snakemake.params["dealer_uids"])),
        )
        transform!(dealer_uids, :transaction_uid => ByRow(UUID) => :transaction_uid)
        transform!(dealer_uids, :dealer_uid => ByRow(readlines) => :dealer_uid)
        dealer_uids = @> dealer_uids flatten(:dealer_uid) transform(
            :dealer_uid => ByRow(UUID) => :dealer_uid
        )
        innerjoin(transactions, dealer_uids, on = :uid => :transaction_uid)
    else
        transactions.dealer_uid .= nothing
        transactions
    end

    open(Arrow.Writer, snakemake.output[1]) do out_writer
        for r in eachrow(iter)
            treated_registrations = Empirics.treated_registrations(r.t; r.dealer_uid)
            if ismissing(treated_registrations)
                @warn "No treated registrations found for transaction $(r.uid)"
                continue
            end
            treated_registrations.D .= true
            Arrow.write(out_writer, treated_registrations)
        end
    end
end
