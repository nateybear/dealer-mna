using DealerMnA
using Splines2
using FixedEffectModels: Term

!isempty(ARGS) && snakemake!(ARGS[1])

function all_registrations(group_df)
    all_arrow_files = map(eachrow(group_df)) do r
        [
            "data/treated/$(r.group)/last_registrations_before_$(r.transaction_uid)_$(r.dealer_uid).arrow",
            "data/treated/$(r.group)/first_registrations_after_$(r.transaction_uid)_$(r.dealer_uid).arrow",
            "data/control/$(r.group)/last_registrations_before_$(r.transaction_uid)_$(r.dealer_uid).arrow",
            "data/control/$(r.group)/first_registrations_after_$(r.transaction_uid)_$(r.dealer_uid).arrow",
        ]
    end
    return DataFrame(Arrow.Table(collect(Iterators.flatten(all_arrow_files))))
end

function splines(x)
    return ns(
        collect(x),
        knots = [-0.1, -0.05, -0.01, 0, 0.01, 0.05, 0.1],
        intercept = true,
    )
end
function addsplines!(df, col = :change_market_share)
    S = splines(df[!, col])
    for i in 1:size(S, 2)
        df[:, "ns_$(i)"] = S[:, i]
    end
    return nothing
end

with_logging() do
    df = DataFrame(Arrow.Table(snakemake.input["group_market_share_changes"]))
    registrations = all_registrations(df)
    dealer_demographics = DataFrame(Arrow.Table(snakemake.input["dealer_demographics"]))
    df.reg_uid = df.transaction_uid .* "_" .* df.dealer_uid
    df = transform(df, [:dealer_uid, :transaction_uid] .=> ByRow(UUID), renamecols=false)
    addsplines!(df)
    heterogeneity_term = mapreduce(Term ∘ Symbol, +, names(df, r"^ns_"))
    extras, coeftable = Empirics.event_study(
        registrations,
        dealer_demographics,
        heterogeneity_term,
        df,
        [:dealer_uid, :reg_uid]
    )
    JLD2.save_object(snakemake.output["extra_stats"], extras)
    Arrow.write(snakemake.output["treatment_effects"], coeftable)
end
