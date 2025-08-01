using LinearAlgebra
using DataFrames
using JLD2
using Arrow

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
    V = only(eachrow(subset(extras, :stat => ByRow(==(column)), view = true))).V
    V_sub = V[[2, 6], [2, 6]] # will always be the same for these columns
    estimate = t_post[column] - t_pre[column]
    se = sqrt(dot([-1, 1], V_sub * [-1, 1]))
    return estimate, se
end
