using DealerMnA

import JLD2
import CSV
using DataFrames

function (@main)(_)
    with_logging() do
        Dealers.load_from_disk!(snakemake.input["dealers_init"])
        transactions = JLD2.load_object(snakemake.input["transactions_for_analysis"])
        out = DataFrame(
            transaction_uid=[t.uid for t in transactions],
            dealer_uid=[
                sort([Dealers.State.df.uid[i] for i in is])
                for is in Dealers.dealer_idxs.(transactions)
            ]
        )
        out = flatten(out, :dealer_uid)
        CSV.write(snakemake.output[1], out)
    end
    return nothing
end
