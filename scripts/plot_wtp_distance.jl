using DealerMnA
using Plots
using StatsBase: mean, median, quantile, trim

function (@main)(_)
    snakemake!()
    with_logging() do
        model = Demand.DemandModel(;
            JLD2.load_object(snakemake.input["model_frames"])...
        )
        params = Float32.(JLD2.load_object(snakemake.input["demand_estimates"]).minimizer)

        # Counterfactual pulls β_price out of the δ-decomposition JLD2 the same
        # way the rest of the codebase does — no need to re-derive.
        cf = Demand.Counterfactual(
            model, params; decomp_path = snakemake.input["delta_decomp"]
        )

        β_price = cf.β_price
        Πp = cf.Πp
        η1, η2 = cf.η[1], cf.η[2]
        dist_scale = model.dist_scale
        price_scale = model.price_scale
        Δd_std = 1.0f0 / dist_scale

        @info "WTP constants" β_price η1 η2 dist_scale price_scale Δd_std

        # α_z per zip (marginal utility of standardized price). Matches the
        # expression inside share_matrix at src/demand/dense.jl:1315-1319.
        z_lookup = Dict(z => i for (i, z) in enumerate(model.zips))
        Dn, Z = size(model.D_dense)
        α_by_zip = Vector{Float32}(undef, Z)
        @inbounds for z in 1:Z
            s = 0.0f0
            for d in 1:Dn
                s += Πp[d] * model.D_dense[d, z]
            end
            α_by_zip[z] = β_price + s
        end
        # Sanity: α should be strictly negative if the estimated model is well-behaved.
        n_nonneg_α = count(≥(0.0f0), α_by_zip)
        n_nonneg_α > 0 &&
            @warn "Non-negative α_z for some zips; those buyers will be dropped" n_nonneg_α n_zips = Z

        fords = read_arrow(snakemake.input["fords"])
        n_all = nrow(fords)

        dealer_uids_in_model = Set(names(model.distances, 2))
        fords_kept = subset(
            fords,
            :ownrzpcd => ByRow(∈(keys(z_lookup))),
            :dealer_uid => ByRow(∈(dealer_uids_in_model)),
        )
        n_kept = nrow(fords_kept)
        n_dropped = n_all - n_kept
        n_dropped > 0 &&
            @warn "Dropped fords rows outside the demand-model zip/dealer support" n_dropped n_all n_kept

        wtp_dollars_per_mile = Float32[]
        sizehint!(wtp_dollars_per_mile, n_kept)
        for r in eachrow(fords_kept)
            z_idx = z_lookup[r.ownrzpcd]
            α = α_by_zip[z_idx]
            α ≥ 0 && continue
            d_std = Float32(model.distances[r.ownrzpcd, r.dealer_uid])
            E_now = exp(η1 + η2 * d_std)
            E_next = exp(η1 + η2 * (d_std + Δd_std))
            ΔU = -(E_next - E_now)
            ΔP_std = -ΔU / α
            push!(wtp_dollars_per_mile, abs(ΔP_std) * price_scale)
        end

        n_buyers = length(wtp_dollars_per_mile)
        median_dollars_per_mile = median(wtp_dollars_per_mile)
        qs = quantile(wtp_dollars_per_mile, [0.05, 0.25, 0.5, 0.75, 0.95])
        @info "WTP distribution (dollars/mile)" n_buyers median_dollars_per_mile qs mean_ = mean(
            wtp_dollars_per_mile
        )

        # Log-scale x if the top decile / median ratio exceeds 10 (heavy tail).
        heavy = quantile(wtp_dollars_per_mile, 0.9) > 10 * median_dollars_per_mile
        histogram(
            collect(trim(wtp_dollars_per_mile, prop=0.05));
            bins = heavy ? 60 : 50,
            xscale = heavy ? :log10 : :identity,
            xlabel = "WTP (\$/mile)",
            label = nothing,
	    normalize = true,
            color = :steelblue,
            alpha = 0.75,
            size = (800, 600),
        )
        vline!(
            [median_dollars_per_mile];
            color = :black, linewidth = 2, linestyle = :dash, alpha=0.75,
            label = "median = \$$(round(Float64(median_dollars_per_mile); digits = 2))/mi",
        )
        png(snakemake.output["plot"])

        JLD2.save_object(
            snakemake.output["summary"],
            (;
                median_dollars_per_mile = Float64(median_dollars_per_mile),
                n_buyers,
                quantiles = Float64.(qs),
            ),
        )
    end
    return nothing
end
