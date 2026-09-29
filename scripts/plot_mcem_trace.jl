using DealerMnA
using Plots
using StatsBase: mean, quantile
using LaTeXStrings

function (@main)(_)
    snakemake!()
    with_logging() do
        trace = JLD2.load_object(snakemake.input["mcem_trace"])
        pi_trace = trace.pi_trace
        gamma_trace = trace.gamma_trace
        phi_hat_trace = trace.phi_hat_trace
        N_TOTAL = trace.N_TOTAL
        BURN_IN = trace.BURN_IN

        @info "MCEM trace loaded" N_TOTAL BURN_IN size_gamma = size(gamma_trace) size_phi = size(
            phi_hat_trace
        )

        rounds = 1:N_TOTAL
        γ_q10 = [quantile(view(gamma_trace, i, :), 0.1) for i in rounds]
        γ_q50 = [quantile(view(gamma_trace, i, :), 0.5) for i in rounds]
        γ_q90 = [quantile(view(gamma_trace, i, :), 0.9) for i in rounds]
        γ_mean = [mean(view(gamma_trace, i, :)) for i in rounds]
        φ_mean = [mean(view(phi_hat_trace, i, :)) for i in rounds]

        p1 = plot(
            rounds, pi_trace;
            label = L"\pi", color = :black, xlabel = "Round", ylabel = L"\pi",
            title = "π trace",
        )
        vline!(p1, [BURN_IN]; color = :gray, linestyle = :dash, label = "burn-in")

        p2 = plot(
            rounds, γ_q10;
            label = "10%", color = :steelblue,
            xlabel = "Round", ylabel = L"\gamma_{ft}",
            title = "γ quantiles per round",
        )
        plot!(p2, rounds, γ_q50; label = "50%", color = :black)
        plot!(p2, rounds, γ_q90; label = "90%", color = :orange)
        vline!(p2, [BURN_IN]; color = :gray, linestyle = :dash, label = nothing)

        p3 = plot(
            rounds, γ_mean;
            label = L"\overline{\gamma}", color = :steelblue,
            xlabel = "Round", ylabel = L"\overline{\gamma_{ft}}",
            title = "Mean posterior compliance",
        )
        vline!(p3, [BURN_IN]; color = :gray, linestyle = :dash, label = nothing)

        p4 = plot(
            rounds, φ_mean;
            label = L"\overline{\varphi}", color = :darkgreen,
            xlabel = "Round", ylabel = L"\overline{\gamma_{ft}}",
            title = "Mean fitted sales target",
        )
        vline!(p4, [BURN_IN]; color = :gray, linestyle = :dash, label = nothing)

        plt = plot(p1, p2, p3, p4; layout = (2, 2), size = (1200, 800), margin = 5Plots.mm)
        png(plt, snakemake.output["plot"])

        post_burn = (BURN_IN + 1):N_TOTAL
        pi_posterior_mean = mean(pi_trace[post_burn])
        mean_gamma_final = mean(view(gamma_trace, N_TOTAL, :))
        mean_phi_final = mean(view(phi_hat_trace, N_TOTAL, :))

        @info "Post-burn-in summaries" pi_posterior_mean mean_gamma_final mean_phi_final

        return JLD2.save_object(
            snakemake.output["summary"],
            (; pi_posterior_mean, mean_gamma_final, mean_phi_final, N_TOTAL, BURN_IN),
        )
    end
    return nothing
end
