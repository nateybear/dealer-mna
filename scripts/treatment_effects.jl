using DealerMnA
import JLD2
import Arrow
using DataFrames
import Lazy: @>
using UUIDs

!isempty(ARGS) && snakemake!(ARGS[1])

with_logging() do
    Dealers.load_from_disk!(snakemake.input["dealers_init"])

    dealer_uid = UUID(snakemake.wildcards["dealer_uid"])

    franchises = Dealers.State.df.franchises[findfirst(
        ==(dealer_uid), Dealers.State.df.uid
    )]

    treated_before = @> snakemake.input["treated_before"] begin
        Arrow.Table
        DataFrame
        subset(:dealer_uid => ByRow(==(dealer_uid)))
    end

    treated_after = @> snakemake.input["treated_after"] begin
        Arrow.Table
        DataFrame
        subset(:dealer_uid => ByRow(==(dealer_uid)))
    end

    control_before = @> snakemake.input["control_before"] begin
        Arrow.Table
        DataFrame
        subset(:make => ByRow(in(franchises)))
    end

    control_after = @> snakemake.input["control_after"] begin
        Arrow.Table
        DataFrame
        subset(:make => ByRow(in(franchises)))
    end

    dealer_demographics = DataFrame(Arrow.Table(snakemake.input["dealer_demographics"]))

    if any(isempty, (treated_before, treated_after, control_before, control_after))
        @warn "One of the input dataframes is empty. Skipping event study."
        Arrow.write(snakemake.output[1], DataFrame())
        return nothing
    end

    extras, coeftable = Empirics.event_study(
        treated_before,
        treated_after,
        control_before,
        control_after,
        dealer_demographics,
    )

    Arrow.write(
        snakemake.output["treatment_effects"],
        coeftable,
    )

    JLD2.save_object(snakemake.output["extra_stats"], extras)
end
