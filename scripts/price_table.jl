using DealerMnA
import JLD2

function (@main)(_)
    with_logging() do
        JLD2.save_object(
            snakemake.output[1],
            Registrations.price_table(),
        )
    end
    return nothing
end
