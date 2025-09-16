using DealerMnA
using StatsBase
using Format
using Base.Iterators

using LaTeXStrings

!isempty(ARGS) && snakemake!(ARGS[1])

process_facts(facts) =
    join(
        map(collect(facts)) do (label, fact)
            "\\newcommand{\\$label}{$fact}"
        end,
        "\n",
    )

with_logging() do
    Dealers.load_from_disk!(snakemake.input["dealers_init"])
    facts = Dict{String, Any}()
    registrations = Registrations.normalize_price(Registrations.registrations())
    transaction_uids = read_csv(snakemake.input["transaction_uids"])
    transactions = JLD2.load_object(snakemake.input["transactions_for_analysis"])
    price_table = Registrations.price_table()
    transform!(
        transaction_uids,
        names(transaction_uids) .=> ByRow(UUID), renamecols = false,
    )

    # HOW MANY REGISTRATIONS?
    push!(facts, "numregistrationstotal" => format(nrow(registrations), commas = true))

    # HOW MANY DEALERS TRANSACTED?
    push!(
        facts,
        "numdealerstransacted" => nrow(transaction_uids),
    )

    # HOW MUCH ARE EXITING SELLERS UNDERPRICING THEIR CARS?
    exiting_sellers = subset(
        transaction_uids,
        :transaction_uid => ByRow(
            t -> countlines("data/affiliated_dealerships/seller_group_$(t).txt") == 0
        ),
    )
    push!(facts, "numexitingsellers" => nrow(exiting_sellers))

    # AVERAGE NUMBER OF CONTROL DEALERS
    push!(
        facts,
        "avgnumcontroldealers" =>
            round(
                Int64,
                mean(eachrow(transaction_uids)) do (; transaction_uid, dealer_uid)
                    control_before = DataFrame(
                        Arrow.Table(
                            "data/control/treated_dealers/last_registrations_before_$(transaction_uid)_$(dealer_uid).arrow"
                        ),
                    )
                    length(unique(control_before.dealer_uid))
                end,
            ),
    )

    # NUBMER OF DUPLICATE/WEIRD GDNS
    multiple_gdns = findall(>(1) ∘ length, Dealers.State.df.gdns)
    shared_gdns = findall(
        .!isempty.(Dealers.State.df.gdns .∩ reshape(Dealers.State.df.gdns, 1, :))
    )
    shared_gdns = Iterators.flatten(idx.I for idx in shared_gdns if !allequal(idx.I))
    push!(facts, "numduplicategdns" => length(multiple_gdns ∪ shared_gdns))

    # NUMBER OF DISTINCT DEALERS
    push!(facts, "numdistinctdealers" => nrow(Dealers.State.df))

    # NUMBER OF DISTINCT GROUPS
    push!(facts, "numdistinctgroups" => length(unique(Dealers.State.df.gid)))

    # TOP FOUR CONCENTRATION
    groups_sorted =
        countmap(Dealers.State.df.gid) |> values |> collect |> sort$(rev = true,)
    push!(
        facts,
        "topfourdealergroupholding" =>
            join(first(groups_sorted, 4), ", ", " and "),
    )
    push!(
        facts,
        "crfourdealergroup" =>
            round(100sum(first(groups_sorted, 4)) / sum(groups_sorted), digits = 2),
    )

    # TRANSACTION SIZE AND WINDOW
    push!(facts, "numtransactionsforanalysis" => length(transactions))
    push!(
        facts,
        "transactionwindowmin" =>
            Dates.format(minimum(t -> t.when, transactions), "U d, Y"),
    )
    push!(
        facts,
        "transactionwindowmax" =>
            Dates.format(maximum(t -> t.when, transactions), "U d, Y"),
    )

    # NUMBER OF TRIMS PER MODEL
    trims_grouped = @> price_table groupby([:make, :model, :model_year]) combine(nrow)
    push!(facts, "numtrimspermodel" => round(mean(trims_grouped.nrow), digits = 2))

    open(snakemake.output[1], "w") do io
        print(io, process_facts(facts))
    end
end
