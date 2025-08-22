using LinearAlgebra
using DataFrames
using JLD2
using Arrow
using Distributions: MvNormal, cdf

function treatment_effect(column, group = "treated_dealers"; transaction_uid, dealer_uid)
    tes = DataFrame(
        Arrow.Table(
            "data/treatment_effects/$(group)/$(transaction_uid)/$(dealer_uid).arrow"
        ),
    )
    extras = JLD2.load_object(
        "data/treatment_effects/$(group)/$(transaction_uid)/$(dealer_uid)_extras.jld2"
    )
    return treatment_effect(tes, extras, column)
end

function treatment_effect(tes, extras, column)
    t_pre = only(eachrow(subset(tes, :T => ByRow(==(-3)), view = true)))
    t_post = only(eachrow(subset(tes, :T => ByRow(==(2)), view = true)))
    D_placebo =
        only(eachrow(subset(extras, :stat => ByRow(==(column)), view = true))).D_placebo
    D_placebo_effect = D_placebo ⋅ [0, -1, 0, 0, 0, 1, 0]
    estimate = t_post[column] - t_pre[column]
    return estimate, D_placebo_effect
end
