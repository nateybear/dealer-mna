using DealerMnA
using Splines2
using FixedEffectModels: Term

!isempty(ARGS) && snakemake!(ARGS[1])

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
    for i in 1:size(S, 2)
        df[:, "ns_$(i)"] = S[:, i]
    end
    return nothing
end

with_logging() do
    df = @> read_arrow(snakemake.input["group_market_share_changes"]) begin
        transform([:dealer_uid, :transaction_uid] .=> ByRow(UUID), renamecols = false)
        subset(:change_market_share => ByRow(!iszero))
        groupby(:dealer_uid)
        subset(:change_market_share => (x -> abs.(x) .== maximum(abs, x)))
    end

    groups = ["treated_buyer", "treated_seller", "control_buyer", "control_seller"]
    registrations = read_arrow(getindex.(Ref(snakemake.input), groups))
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
