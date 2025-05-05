using DealerMnA

import JLD2

function (@main)(_)
    with_logging() do
        transactions = JLD2.load_object(snakemake.input["transactions_for_analysis"])
        open(snakemake.output[1], "w") do io
            for t in transactions
                println(io, t.uid)
            end
        end
    end
end
