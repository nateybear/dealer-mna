using DealerMnA
using .Demand
using DataFrames
using Statistics: median
using Printf
using Distributions: Normal, cdf
using FixedEffectModels: coef, coefnames, stderror
using CSV

const DEMO_LABELS = Dict(
    "median_earnings_inv" => "Inv. Median Earnings",
    "urban_pct" => "Frac. Urban",
    "hispanic_pct" => "Frac. Hispanic",
    "black_pct" => "Frac. Black",
)
const DEMO_ORDER = ["median_earnings_inv", "urban_pct", "hispanic_pct", "black_pct"]
const PRODUCT_LABELS = Dict(
    "is_coupe" => "Coupe",
    "is_suv" => "SUV",
    "is_pickup" => "Pickup",
    "is_sedan" => "Sedan",
)
const PRODUCT_ORDER = ["is_coupe", "is_suv", "is_pickup", "is_sedan"]

function stars(p)
    isnan(p) && return ""
    p < 0.01 && return "^{***}"
    p < 0.05 && return "^{**}"
    p < 0.1 && return "^{*}"
    return ""
end

fmt_est(est, p; digits = 3) = @sprintf("\$%.*f%s\$", digits, est, stars(p))
fmt_se(se; digits = 3) = @sprintf("\$(%.*f)\$", digits, se)

function median_own_price_elasticity(model, params)
    work = Demand.DemandWork(model)
    Demand.forward_pass!(work, model, params)

    idx = Demand.param_indices(model)
    Πp = view(params, idx.i_Πp)
    decomp = JLD2.load_object("data/delta_decomposition_regout.jld2")
    β_price = Float32(coef(decomp.regout)[findfirst(==("price"), coefnames(decomp.regout))])

    Dn, Z = size(model.D_dense)
    α = Vector{Float32}(undef, Z)
    @inbounds for z in 1:Z
        s = 0.0f0
        for d in 1:Dn
            s += Πp[d] * model.D_dense[d, z]
        end
        α[z] = β_price + s
    end
    M = model.M_zs
    price_scale = model.price_scale

    eps = Float32[]
    sizehint!(eps, sum(size(U, 1) for U in work.U_blocks))
    for bi in eachindex(work.U_blocks)
        U = work.U_blocks[bi]
        Nb, Zb = size(U)
        for k in 1:Nb
            global_idx = model.block_param_idxs[bi][k]
            p_raw = Float32(model.product_info[global_idx].p) * price_scale
            q = 0.0f0
            dq = 0.0f0
            @inbounds for z in 1:Zb
                s_jz = U[k, z]
                q += M[z] * s_jz
                dq += M[z] * s_jz * (1 - s_jz) * α[z]
            end
            q > 0 || continue
            push!(eps, Float32(dq * p_raw / (price_scale * q)))
        end
    end
    return Float64(median(eps)), length(eps)
end

function (@main)(_)
    snakemake!()
    with_logging() do
        model = Demand.DemandModel(;
            JLD2.load_object(snakemake.input["model_frames"])...
        )
        params = Float32.(JLD2.load_object(snakemake.input["demand_estimates"]).minimizer)

        se_df = CSV.read(snakemake.input["demand_se"], DataFrame)
        se_lookup = Dict(zip(se_df.name, zip(se_df.estimate, se_df.std_error, se_df.p_val)))

        decomp = JLD2.load_object(snakemake.input["delta_decomp"])
        regout = decomp.regout
        names_r = coefnames(regout)
        coefs_r = coef(regout)
        ses_r = stderror(regout)
        reg_lookup = Dict(n => (coefs_r[i], ses_r[i]) for (i, n) in enumerate(names_r))
        reg_pval(est, se) = 2 * cdf(Normal(), -abs(est / se))

        med_eps, n_eps = median_own_price_elasticity(model, params)
        @info "Median own-price elasticity" median = med_eps n = n_eps

        io = IOBuffer()
        println(io, "\\begin{tabular}{l|cccc}")
        header = join([DEMO_LABELS[d] for d in DEMO_ORDER], " & ")
        println(io, "     & ", header, " \\\\")
        println(io, "     \\toprule")
        for p in PRODUCT_ORDER
            est_row = String[]
            se_row = String[]
            for d in DEMO_ORDER
                key = "Π[$d,$p]"
                est, se, pv = se_lookup[key]
                push!(est_row, fmt_est(est, pv))
                push!(se_row, fmt_se(se))
            end
            println(io, "     ", PRODUCT_LABELS[p], " & ", join(est_row, " & "), " \\\\")
            println(io, "           & ", join(se_row, " & "), " \\\\[0.5em]")
        end
        price_est = String[]
        price_se = String[]
        for d in DEMO_ORDER
            key = "Πp[$d]"
            est, se, pv = se_lookup[key]
            push!(price_est, fmt_est(est, pv))
            push!(price_se, fmt_se(se))
        end
        println(io, "     Price & ", join(price_est, " & "), " \\\\")
        println(io, "           & ", join(price_se, " & "), " \\\\")
        println(io, "     \\midrule")

        β_est, β_se = reg_lookup["price"]
        β_pv = reg_pval(β_est, β_se)
        η1_est, η1_se, η1_pv = se_lookup["η_baseline"]
        println(io,
            "     Mean Price Effect & ", fmt_est(β_est, β_pv),
            " & & \$\\eta_1\$ (Dist. Baseline) & ", fmt_est(η1_est, η1_pv), " \\\\",
        )
        println(io,
            "     & ", fmt_se(β_se), " & & & ", fmt_se(η1_se), " \\\\[0.5em]",
        )

        γ_est, γ_se = reg_lookup["log(J)"]
        γ_pv = reg_pval(γ_est, γ_se)
        η2_est, η2_se, η2_pv = se_lookup["η_dist"]
        println(io,
            "     \$\\log(J)\$ & ", fmt_est(γ_est, γ_pv),
            " & & \$\\eta_2\$ (Distance) & ", fmt_est(η2_est, η2_pv), " \\\\",
        )
        println(io,
            "     & ", fmt_se(γ_se), " & & & ", fmt_se(η2_se), " \\\\[0.5em]",
        )

        println(io,
            "     Median Own-Price Elasticity & ",
            @sprintf("\$%.3f\$", med_eps), " \\\\",
        )
        println(io, "     \\bottomrule")
        println(io, "\\end{tabular}")

        write(snakemake.output["table"], String(take!(io)))
        @info "Wrote demand table" path = snakemake.output["table"]
    end
    return nothing
end
