using DealerMnA

import JLD2

function (@main)(_)
    with_logging() do
        Dealers.load_from_disk!(snakemake.input["dealers_init"])
        out = Dealers.Transaction[]
        for (i, t) in enumerate(Dealers.Events)
            if t isa Dealers.Transaction &&
                !isempty(Dealers.Diffs[i]) &&
                !ismissing(Empirics.blackout_period(t)) &&
                Empirics.blackout_period(t) ⊆ Empirics.ANALYSIS_DATES
                @info "Transaction $(t.uid) is in the analysis period"
                last_registrations = Empirics.last_registrations_before(t)
                first_registrations = Empirics.first_registrations_after(t)
                if !ismissing(last_registrations) &&
                    !ismissing(first_registrations) &&
                    !isempty(last_registrations) &&
                    !isempty(first_registrations)
                    @info "Transaction $(t.uid) has registrations before and after"
                    push!(out, t)
                end
            end
        end
        JLD2.save_object(
            snakemake.output[1],
            out,
        )
    end
    return nothing
end
