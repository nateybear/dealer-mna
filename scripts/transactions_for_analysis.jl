using DealerMnA

!isempty(ARGS) && snakemake!(ARGS[1])

with_logging() do
    Dealers.load_from_disk!(snakemake.input["dealers_init"])
    out = Dealers.Transaction[]
    for (i, t) in enumerate(Dealers.Events)
        if t isa Dealers.Transaction &&
            !isempty(Dealers.Diffs[i]) &&
            !ismissing(Empirics.blackout_period(t)) &&
            Empirics.blackout_period(t) ⊆ Empirics.ANALYSIS_DATES &&
            t.buyer_gid != t.seller_gid
            @info "Transaction $(t.uid) is in the analysis period"
            treated_registrations = Empirics.treated_registrations(t)
            controls = Empirics.control_dealers(t)
            if !ismissing(treated_registrations) &&
                treated_registrations.τ ⊇ Empirics.VALID_PERIODS &&
                !isempty(controls)
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
