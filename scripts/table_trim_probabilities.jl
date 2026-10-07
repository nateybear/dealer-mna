using DealerMnA
using XGBoost
using StatsBase: mean, std
using LogExpFunctions
using LaTeXTabulars
using LaTeXStrings
using Printf

function predict_trim(dm, pbase_bytes, rho_bytes)
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

fmt3(x) = @sprintf("%.3f", x)

function (@main)(_)
    snakemake!()
    with_logging() do
        state_matrix = JLD2.load_object(snakemake.input["state_matrix"]).matrix
        fits = JLD2.load_object(snakemake.input["fits"])
        fords = read_arrow(snakemake.input["fords"])

        # kept_trims and model_year ≤ 2019 are applied upstream in load_model_frames.

        trim_counts = @> fords groupby([:model, :model_year, :extra]) combine(nrow => :n_sales)
        n_sales_lookup = Dict(
            (r.model, r.model_year, r.extra) => r.n_sales for r in eachrow(trim_counts)
        )

        n_dealer_quarters = size(state_matrix, 1)
        dm = XGBoost.DMatrix(state_matrix)

        table = DataFrame(
            model = String[], model_year = Int64[], extra = String[],
            n_sales = Int[], n_dealer_quarters = Int[],
            mean_pbase = Float32[], std_pbase = Float32[],
            mean_rho = Float32[], std_rho = Float32[],
            mean_p_carry = Float32[],
        )
        for (k, v) in pairs(fits)
            haskey(n_sales_lookup, k) || continue
            p_base, ρ̃ = predict_trim(dm, v[1], v[2])
            p_carry = p_base .+ (1 .- p_base) .* ρ̃
            push!(
                table,
                (
                    k[1], k[2], k[3],
                    n_sales_lookup[k], n_dealer_quarters,
                    mean(p_base), std(p_base),
                    mean(ρ̃), std(ρ̃),
                    mean(p_carry),
                ),
            )
        end
        finalize(dm)
        sort!(table, [:model, :model_year, :extra])

        @info "Trim table" nrows = nrow(table)

        CSV.write(snakemake.output["csv"], table)

        header = [
            "Model", "MY", "Trim",
            L"n_{\text{sales}}", L"n_{d,t}",
            L"\overline{p}_{\text{base}}", L"\sigma(p_{\text{base}})",
            L"\overline{\tilde\rho}", L"\sigma(\tilde\rho)",
            L"\overline{P(\text{carry}\mid\text{comply})}",
        ]
        row_fmt(r) = [
            r.model, string(r.model_year), r.extra,
            string(r.n_sales), string(r.n_dealer_quarters),
            fmt3(r.mean_pbase), fmt3(r.std_pbase),
            fmt3(r.mean_rho), fmt3(r.std_rho),
            fmt3(r.mean_p_carry),
        ]
        open(snakemake.output["tex"], "w") do io
            return latex_tabular(
                io,
                Tabular("lll" * "r"^7),
                [
                    Rule(:top),
                    header,
                    Rule(:mid),
                    row_fmt.(eachrow(table))...,
                    Rule(:bottom),
                ],
            )
        end
    end
    return nothing
end
