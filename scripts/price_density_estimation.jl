using Distributed

@everywhere using Memoize

@everywhere begin
    using DealerMnA
    using .Demand
    using XGBoost
    using LoggingExtras

    model = DemandModel(; JLD2.load_object("data/model_frames.jld2")...)
    state_matrix = JLD2.load_object("data/residual_demand.jld2")

    @memoize function j_to_tf_lookup()
        base = Supply.tf_to_j_lookup(model)
        idx_dict = Dict(k => i for (i, k) in enumerate(keys(base)))
        out = Dict{VehicleKey, Vector{Tuple{Tuple{QuarterKey, UUID}, Int64}}}()
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

    # OOS-best-round + full-data refit. Identity link via reg:squarederror; mean
    # NLL is MSE which Supply.fit_xgb_oos uses as the default val_loss.
    regress(X, y) =
        Supply.fit_xgb_oos(
            X,
            y;
            max_round = 1000,
            patience = 5,
            params = (subsample = 0.8,),
        )

    function fit_prices(j)
        @info "Fitting" j
        X, y = X_matrix(j), log_prices(j)
        mean_booster, mean_round = regress(X, y)
        y_hat = XGBoost.predict(mean_booster, X)
        u = abs.(y .- y_hat)
        stddev_booster, stddev_round = regress(X, u)
        @info "  best rounds" j mean_round stddev_round
        return XGBoost.serialize(mean_booster), XGBoost.serialize(stddev_booster)
    end
end

function (@main)(_)
    js = collect(keys(model.product_lookup.j))
    fitted = pmap(fit_prices, js)
    JLD2.save_object("data/price_densities.jld2", Dict(js .=> fitted))
    return nothing
end
