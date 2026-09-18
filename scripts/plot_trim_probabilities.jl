using DealerMnA
using Plots
using StatsPlots
using XGBoost
using StatsBase: mean, std, quantile
using LogExpFunctions

# Predict (p_base, ρ̃) on the full state matrix for one trim.
function predict_trim(dm::XGBoost.DMatrix, pbase_bytes, rho_bytes)
    booster = XGBoost.Booster(dm)
    XGBoost.load!(booster, pbase_bytes)
    f_raw = XGBoost.predict(booster, dm; margin = true)
    p_base = logistic.(f_raw)
    if isempty(rho_bytes)
        ρ̃ = fill(Float32(Supply.RHO_MIN), length(p_base))
    else
        XGBoost.load!(booster, rho_bytes)
        g_raw = XGBoost.predict(booster, dm; margin = true)
        ρ̃ = Supply.rho_from_margin.(g_raw)
    end
    finalize(booster)
    return Float32.(p_base), Float32.(ρ̃)
end

function trim_panel(model_name, model_year, extra, n_sales, p_base, ρ̃)
    p_carry = p_base .+ (1 .- p_base) .* ρ̃
    return groupedhist(
        vcat(p_base, p_carry);
        bar_position = :stack,
        group = vcat(
            repeat(["p_base"], length(p_base)),
            repeat(["P(carry|comply)"], length(p_carry)),
        ),
        fill = [:steelblue :orange],
        xlims = (0, 1),
        title = "$model_name $model_year $extra (n=$n_sales)",
        titlefont = font(8),
        xtickfontsize = 6,
        ytickfontsize = 6,
        legendfontsize = 6,
        bins = 30,
        yticks = false,
    )
end

function (@main)(_)
    snakemake!()
    with_logging() do
        state_matrix = JLD2.load_object(snakemake.input["state_matrix"])
        fits = JLD2.load_object(snakemake.input["fits"])

        # State DMatrix is shared across all trims (per-car boosters use only
        # state_matrix rows — see X_matrix in compliance_estimation.jl:95-106).
        dm = XGBoost.DMatrix(state_matrix)

        fords = read_arrow(snakemake.input["fords"])
        # Match compliance_estimation.jl's universe.
        kept_pairs = Set(
            (row.model, row.extra) for
            row in eachrow(read_arrow(snakemake.input["kept_trims"]))
        )
        fords = subset(fords, [:model, :extra] => ByRow((m, e) -> (m, e) ∈ kept_pairs))
        fords = subset(fords, :model_year => ByRow(≤(2019)))

        trim_counts = @> fords groupby([:model, :model_year, :extra]) combine(nrow => :n_sales)

        keys_all = collect(keys(fits))
        n_sales_lookup = Dict(
            (r.model, r.model_year, r.extra) => r.n_sales for r in eachrow(trim_counts)
        )
        keys_all = [k for k in keys_all if haskey(n_sales_lookup, k)]

        # Predict once per trim; store summaries + arrays for the featured trims.
        pred_cache = Dict{Tuple{String, Int64, String}, Tuple{Vector{Float32}, Vector{Float32}}}()
        summary = DataFrame(
            model = String[], model_year = Int64[], extra = String[],
            n_sales = Int[],
            mean_pbase = Float32[], std_pbase = Float32[],
            mean_rho = Float32[], std_rho = Float32[],
        )
        for k in keys_all
            pbase_bytes, rho_bytes = fits[k][1], fits[k][2]
            p_base, ρ̃ = predict_trim(dm, pbase_bytes, rho_bytes)
            pred_cache[k] = (p_base, ρ̃)
            push!(
                summary,
                (
                    k[1], k[2], k[3], n_sales_lookup[k],
                    mean(p_base), std(p_base), mean(ρ̃), std(ρ̃),
                ),
            )
        end
        finalize(dm)

        sort!(summary, :n_sales, rev = true)
        popular_keys = [
            (r.model, r.model_year, r.extra) for r in eachrow(first(summary, 6))
        ]

        n_total = nrow(summary)
        lo_cut = quantile(summary.n_sales, 1 / 3)
        hi_cut = quantile(summary.n_sales, 2 / 3)
        mid = subset(
            summary,
            :n_sales => ByRow(x -> lo_cut ≤ x ≤ hi_cut),
        )
        sort!(mid, :std_rho, rev = true)
        variation_keys = [
            (r.model, r.model_year, r.extra) for r in eachrow(first(mid, 6))
        ]

        @info "Featured trims" n_total popular_keys variation_keys

        function panel_for(k)
            p_base, ρ̃ = pred_cache[k]
            return trim_panel(k[1], k[2], k[3], n_sales_lookup[k], p_base, ρ̃)
        end
        panels = vcat(panel_for.(popular_keys), panel_for.(variation_keys))
        plt = plot(panels...; layout = (4, 3), size = (1200, 900),
            plot_title = "Top 6 popular (rows 1–2) vs. mid-popularity, top std(ρ̃) (rows 3–4)",
            plot_titlefont = font(10),
        )
        return png(plt, snakemake.output["plot"])
    end
    return nothing
end
