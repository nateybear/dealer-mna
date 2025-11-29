using DealerMnA
using Combinatorics

!isempty(ARGS) && snakemake!(ARGS[1])

with_logging() do
    Dealers.load_from_disk!(snakemake.input["dealers_init"])
    out = Dealers.Transaction[]
    dealer_demographics = read_arrow(snakemake.input["dealer_demographics"])
    for (i, t) in enumerate(Dealers.Events)
        if t isa Dealers.Transaction &&
            !isempty(Dealers.Diffs[i]) &&
            !ismissing(Empirics.blackout_period(t)) &&
            Empirics.blackout_period(t) ⊆ Empirics.ANALYSIS_DATES &&
            t.buyer_gid != t.seller_gid
            @info "Transaction $(t.uid) is in the analysis period"
            treated_registrations = Empirics.treated_registrations(t)
            controls = Empirics.control_dealers(t; dealer_demographics)
            if !ismissing(treated_registrations) &&
                !isempty(controls) &&
                treated_registrations.τ ⊇ Empirics.VALID_PERIODS
                @info "Transaction $(t.uid) has registrations before and after"
                push!(out, t)
            end
        end
    end

    # find multiple transactions on one dealership
    doubled_transactions = Set{UUID}()
    for (i, j) in combinations(out, 2)
        if !isempty(
            intersect(
                Dealers.State.df.uid[Dealers.dealer_idxs(i)],
                Dealers.State.df.uid[Dealers.dealer_idxs(j)],
            ),
        )
            push!(doubled_transactions, i.uid)
            push!(doubled_transactions, j.uid)
        end
    end

    @info "Removing $(length(doubled_transactions)) transactions that share a dealership with another transaction" doubled_transactions

    out = [ t for t in out if !(t.uid in doubled_transactions) ]

    JLD2.save_object(
        snakemake.output[1],
        out,
    )
end
