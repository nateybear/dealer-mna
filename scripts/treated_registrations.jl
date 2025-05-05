using DealerMnA
using DrWatson
import JLD2
import Arrow

function (@main)(_)
    with_logging() do
        Dealers.load_from_disk!(snakemake.input["dealers_init"])
        transactions = JLD2.load_object(snakemake.input["transactions_for_analysis"])
        for t in transactions
            @info "Writing treated registrations for transaction $(t.uid)"
            Arrow.write(
                datadir("treated_registrations/last_registrations_before_$(t.uid).arrow"),
                Empirics.last_registrations_before(t),
            )
            Arrow.write(
                datadir("treated_registrations/first_registrations_after_$(t.uid).arrow"),
                Empirics.first_registrations_after(t),
            )
        end
    end
    return nothing
end
