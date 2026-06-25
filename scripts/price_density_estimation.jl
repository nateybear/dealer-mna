using Distributed

@everywhere using Memoize

@everywhere begin
    using DealerMnA
    using .Demand
    using XGBoost

    model = DemandModel(; JLD2.load_object("data/model_frames.jld2")...)
    state_matrix = JLD2.load_object("data/residual_demand.jld2")

    @memoize function j_to_tf_lookup()
        base = Supply.tf_to_j_lookup(model)
        idx_dict = Dict(k => i for (i, k) in enumerate(keys(base)))
        out = Dict{VehicleKey, Vector{Int64}}()
        for (tf, js) in base
            for j in js
                if haskey(out, j)
                    push!(out[j], (tf, idx_dict[tf]))
                else
                    out[j] = [(tf, idx_dict[tf])]
                end
            end
        end
        return out
    end

    X_matrix(j) =
        mapreduce(vcat, j_to_tf_lookup()[j]) do (_, i)
            reshape(state_matrix[i, :], 1, :)
        end

    log_prices(j) =
        map(j_to_tf_lookup()[j]) do (tf, _)
            t, f = tf
            idx = model.product_lookup.tjf[(t, j, f)]
            return log(model.product_info[idx].p)
        end

    function fit_prices(j)
        X, y = X_matrix(j, model, state_matrix), log_prices(j, model)
        mean_booster = xgboost((X, y), num_rounds = 15, XGBoost.regression()...)
        y_hat = XGBoost.predict(mean_booster, X)
        u = abs.(y - y_hat)
        stddev_booster = xgboost((X, u), num_rounds = 15, XGBoost.regression()...)
        return XGBoost.serialize(mean_booster), XGBoost.serialize(stddev_booster)
    end
end

function (@main)(_)
    js = keys(model.product_lookup.j)
    fitted = pmap(fit_prices, js)
    JLD2.save_object("data/price_densities.jld2", Dict(js .=> fitted))
    return nothing
end
