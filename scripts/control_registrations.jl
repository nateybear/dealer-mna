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

    dealer_demographics = DataFrame(Arrow.Table(snakemake.input["dealer_demographics"]))

    iter = if haskey(snakemake.input, "dealer_uids")
        dealer_uids = DataFrame(
            transaction_uid = collect(keys(snakemake.input["dealer_uids"])),
            dealer_uid = collect(values(snakemake.input["dealer_uids"])),
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

    controls = Empirics.iter_control_dealers(
        collect(zip(iter.t, iter.dealer_uid)); dealer_demographics
    )

    open(Arrow.Writer, snakemake.output[1]) do out_writer
        for r in controls
            control_regs = Empirics.registrations_in_window(
                r.window; r.dealer_uids, r.franchises
            )
            control_regs.τ .= 0
            control_regs.D .= 0
            Arrow.write(
                out_writer,
                control_regs,
            )
        end
    end
end
