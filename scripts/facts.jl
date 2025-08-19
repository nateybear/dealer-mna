using DealerMnA
using StatsBase
using Format

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
    transform!(
        transaction_uids,
        names(transaction_uids) .=> ByRow(UUID), renamecols = false,
    )

    # HOW MANY REGISTRATIONS?
    push!(facts, "numregistrationstotal" => format(nrow(registrations), commas = true))

    # HOW MUCH ARE EXITING SELLERS UNDERPRICING THEIR CARS?
    exiting_sellers = subset(
        transaction_uids,
        :transaction_uid => ByRow(
            t -> countlines("data/affiliated_dealerships/seller_group_$(t).txt") == 0
        ),
    )
    push!(
        facts,
        "exitingsellerunderprice" =>
            round(
                Int64,
                mean(eachrow(exiting_sellers)) do (; transaction_uid, dealer_uid)
                    tes = DataFrame(
                        Arrow.Table(
                            "data/treatment_effects/treated_dealers/$(transaction_uid)/$(dealer_uid).arrow"
                        ),
                    )
                    extras = JLD2.load_object(
                        "data/treatment_effects/treated_dealers/$(transaction_uid)/$(dealer_uid)_extras.jld2"
                    )
                    -first(Empirics.synthetic_comparison(tes, extras, :mean_price, -4:-3))
                end,
            ),
    )

    open(snakemake.output[1], "w") do io
        print(io, process_facts(facts))
    end
end
