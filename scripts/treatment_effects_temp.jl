using DealerMnA
import JLD2
import Arrow
using DataFrames
import Lazy: @>
using UUIDs
using StatsBase: median
using FixedEffectModels: Term, ConstantTerm

!isempty(ARGS) && snakemake!(ARGS[1])

with_logging() do
    Dealers.load_from_disk!(snakemake.input["dealers_init"])
    registrations = read_arrow([
        snakemake.input["treated_registrations"],
        snakemake.input["control_registrations"],
    ])
    transactions = JLD2.load_object(snakemake.input["transactions_for_analysis"])
    cstimetable = flatten(
        DataFrame(
            (
                dealer_uid = Dealers.State.df.uid[Dealers.dealer_idxs(t)],
                event_time = Dealers.cstime(t),
            ) for t in transactions
        ),
        :dealer_uid,
    )
    leftjoin!(registrations, cstimetable, on = :dealer_uid)
    registrations.event_time = coalesce.(registrations.event_time, 0)
    registrations.cstime = Dealers.cstime.(registrations.t)
    transaction_uids = read_csv(snakemake.input["transaction_uids"])
    mapcols!(ByRow(UUID), transaction_uids)
    dealer_demographics = DataFrame(Arrow.Table(snakemake.input["dealer_demographics"]))



    Arrow.write(
        snakemake.output["treatment_effects"],
        coeftable,
    )

    JLD2.save_object(snakemake.output["extra_stats"], extras)
end
