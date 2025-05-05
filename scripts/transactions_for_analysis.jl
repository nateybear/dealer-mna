using DealerMnA

import JLD2

function (@main)(_)
    with_logging() do
        Dealers.load_from_disk!(snakemake.input["dealers_init"])
        transactions_for_analysis = [
            t for (i, t) in enumerate(Dealers.Events)
            if t isa Dealers.Transaction &&
            !isempty(Dealers.Diffs[i]) &&
            !ismissing(Empirics.blackout_period(t)) &&
            Empirics.blackout_period(t) ⊆ Empirics.ANALYSIS_DATES &&
            !isempty(Empirics.last_registrations_before(t)) &&
            !isempty(Empirics.first_registrations_after(t))
        ]
        JLD2.save_object(
            snakemake.output[1],
            transactions_for_analysis
        )
    end
    return nothing
end
