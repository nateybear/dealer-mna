using DealerMnA
using Plots
using XGBoost
using StatsBase: mean, std, corspearman
using LogExpFunctions
using LaTeXStrings

function (@main)(_)
    snakemake!()
    with_logging() do
        demand_model = Demand.DemandModel(;
            JLD2.load_object(snakemake.input["model_frames"])...
        )
        _rd = JLD2.load_object(snakemake.input["state_matrix"])
        state_matrix = _rd.matrix
        state_lookup = Dict(k => i for (i, k) in enumerate(_rd.tf_keys))

        compliance = read_arrow(snakemake.input["compliance"])
        fords = read_arrow(snakemake.input["fords"])
        # kept_trims applied upstream in load_model_frames.
        fords = subset(fords, :model_year => ByRow(≤(2019)))

        dq = select(Supply.make_dealer_quarter(fords), :dealer_uid, :t, :total_units => :q)

        # ι_lag: p_compliant at (dealer, t−1) > 0.5. Mirrors sales_target_estimation.jl:48-52.
        compliance_lag = select(compliance, [:dealer_uid, :t, :p_compliant])
        transform!(compliance_lag, :t => ByRow(x -> x + Quarter(1)) => :t)
        transform!(compliance_lag, :p_compliant => ByRow(>(0.5)) => :ι_lag)
        select!(compliance_lag, Not(:p_compliant))

        panel = innerjoin(compliance, dq, on = [:dealer_uid, :t])
        panel = innerjoin(panel, compliance_lag, on = [:dealer_uid, :t])

        n_before = nrow(panel)
        panel = subset(
            panel,
            [:t, :dealer_uid] => ByRow(
                (t, f) -> haskey(
                    state_lookup,
                    (
                        (Dates.year(t - Quarter(1)), Dates.quarterofyear(t - Quarter(1))),
                        f,
                    ),
                ),
            ),
        )
        n_after = nrow(panel)
        n_dropped = n_before - n_after
        n_dropped > 0 && @warn "Dropped (t, f) rows with missing lagged state row" n_dropped n_before n_after

        n = nrow(panel)
        ncols_state = size(state_matrix, 2)
        # Layout: [state_matrix[t-1], quarter_of_year(t), ι_lag]. Must match
        # compliance_estimation.jl's X_agg column order.
        X = Matrix{Float32}(undef, n, ncols_state + 2)
        for (i, r) in enumerate(eachrow(panel))
            tlag = r.t - Quarter(1)
            quarter = (Dates.year(tlag), Dates.quarterofyear(tlag))
            X[i, 1:ncols_state] .= @view state_matrix[state_lookup[(quarter, r.dealer_uid)], :]
            X[i, ncols_state + 1] = Float32(Dates.quarterofyear(r.t))
            X[i, end] = Float32(r.ι_lag)
        end

        booster_bytes, τ = JLD2.load_object(snakemake.input["sales_targets"])
        dm = XGBoost.DMatrix(X)
        booster = XGBoost.Booster(dm)
        XGBoost.load!(booster, booster_bytes)
        φ̂ = exp.(XGBoost.predict(booster, dm; margin = true))
        finalize(booster)
        finalize(dm)

        q = Float32.(panel.q)
        p_compliant = Float32.(panel.p_compliant)

        ρ_spearman = corspearman(Float64.(q), Float64.(φ̂))
        @info "Sales-target scatter" n τ ρ_spearman mean_q = mean(q) mean_φ̂ = mean(φ̂) mean_p_compliant = mean(
            p_compliant
        )

        xy_max = max(maximum(q), maximum(φ̂))
        xy_min = max(1.0f1, min(minimum(q), minimum(φ̂)))
        scatter(
            q, φ̂;
            marker_z = p_compliant,
            color = :viridis,
            clims = (0.0, 1.0),
            colorbar_title = "P(ι = 1 | data)",
            xscale = :log10, yscale = :log10,
            xlims = (xy_min, xy_max), ylims = (xy_min, xy_max),
            xlabel = L"q_{f,t}",
            ylabel = L"\hat\varphi_{f,t}",
            markersize = 2.5, markerstrokewidth = 0, alpha = 0.6,
            label = nothing,
            size = (700, 600),
        )
        plot!([xy_min, xy_max], [xy_min, xy_max]; color = :red, linestyle = :dash, label = "y = x")
        png(snakemake.output["plot"])
    end
    return nothing
end
