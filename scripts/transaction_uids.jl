using DealerMnA

import JLD2
import CSV
using DataFrames

function (@main)(_)
    with_logging() do
        Dealers.load_from_disk!(snakemake.input["dealers_init"])
        transactions = JLD2.load_object(snakemake.input["transactions_for_analysis"])
        out = DataFrame()
        for t in transactions
            @info "Processing transaction $(t.uid)"
            before = Empirics.last_registrations_before(t)
            after = Empirics.first_registrations_after(t)
            uids = intersect(before.dealer_uid, after.dealer_uid)
            @info "Added $(length(uids)) dealers"
            append!(out, DataFrame(transaction_uid = t.uid, dealer_uid = sort(uids)))
        end
        CSV.write(snakemake.output[1], out)
    end
    return nothing
end
