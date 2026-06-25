using DealerMnA

using .Demand

model = DemandModel(; JLD2.load_object("data/model_frames.jld2")...)
params = JLD2.load_object("data/demand_estimates.jld2").minimizer

cf = Counterfactual(model, params)

iter = Supply.tf_to_j_lookup(cf)

J = length(keys(model.product_lookup.j))

out = Matrix{Float32}(undef, length(keys(iter)), J * (J + 1))

for (i, (t, f)) in enumerate(keys(iter))
    @info "Writing dealer" i t f
    out[i, :] = reshape(Supply.residual_demand(cf, f, t), :)
end

JLD2.save_object("data/residual_demand.jld2", out)
