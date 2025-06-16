function synthetic_comparison(
    tes,
    extra,
    column,
    periods = :all,
)
    extra = only(
        eachrow(subset(extra, :stat => ByRow(==(column)), view = true))
    )

    V = extra.V
    Δ = extra.fe_treated - extra.fe_synthetic

    periods == :all && (periods = VALID_PERIODS)
    N = length(periods)
    ∇ = (tes.T .∈ Ref(periods)) / N

    estimate = sum(∇ .* tes[!, column])
    se = sqrt(dot(∇, V * ∇))
    estimate += Δ

    return estimate, se
end
