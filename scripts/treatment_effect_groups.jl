using DealerMnA
using Splines2
using FixedEffectModels: Term

snakemake!()

const NS = Ref{Function}()

function calcsplines!(x)
    if !isdefined(NS, 1)
        NS[] = ns_(
            collect(x),
            boundary_knots = (-0.1, 0.1),
            df = 5,
            intercept = true,
        )
    end
    return NS[]
end

function addsplines!(df, col = :change_market_share)
    f = calcsplines!(df[!, col])
    S = f(df[!, col])
    for i in axes(S, 2)
        df[:, "ns_$(i)"] = S[:, i]
    end
    return nothing
end

with_logging() do
    df = @> read_arrow(snakemake.input["group_market_share_changes"]) begin
        transform([:dealer_uid, :transaction_uid] .=> ByRow(UUID), renamecols = false)
        groupby(:dealer_uid)
        subset(:change_market_share => (x -> abs.(x) .== maximum(abs, x)))
        unique(:dealer_uid) # if multiple maxima select the first
    end

    transactions = JLD2.load_object(snakemake.input["transactions_for_analysis"])
    transactions = DataFrame(
        uid = [t.uid for t in transactions],
        t = transactions,
    )

    registrations = read_arrow([snakemake.input["treated"], snakemake.input["control"]])
    # remove registrations that we filtered out above
    registrations = subset(
        registrations,
        [:D, :dealer_uid] => ByRow((d, duid) -> !d || in(duid, df.dealer_uid)),
        view = true,
    )

    # registrations may have multiple τ at a give t because of how df is calculated
    retained_τ_table = mapreduce(vcat, eachrow(df)) do r
        transaction = pluck(transactions, :uid => ByRow(==(r.transaction_uid))).t
        return DataFrame(
            dealer_uid = r.dealer_uid,
            τ = collect(-4:3),
            t = Dealers.eventtime.(Ref(transaction), -4:3),
        )
    end
    registrations = vcat(
        semijoin(# treated registrations
            subset(registrations, :D),
            retained_τ_table,
            on = [:dealer_uid, :t, :τ],
        ),
        antijoin(# control registrations
            subset(registrations, :D => ByRow(!)),
            retained_τ_table,
            on = :dealer_uid,
        ),
    )
    dealer_demographics = read_arrow(snakemake.input["dealer_demographics"])

    addsplines!(df)
    heterogeneity_term = mapreduce(Term ∘ Symbol, +, names(df, r"^ns_"))

    config = let Config = Empirics.parse_event_study_config(snakemake.wildcards["specification"])
        Config(
            registrations = registrations,
            dealer_demographics = dealer_demographics,
            interaction_style = Empirics.WithAdditionalInteraction(
                term = heterogeneity_term,
                additional_data = df,
            ),
        )
    end

    JLD2.save_object(snakemake.output[1], Empirics.event_study(config))
end
