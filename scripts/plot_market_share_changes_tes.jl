using DealerMnA
using Plots
using StatsBase
using Loess

!isempty(ARGS) && snakemake!(ARGS[1])

function extract_tes(col, tuid, duid, group)
    tes = DataFrame(Arrow.Table("data/treatment_effects/$(group)/$(tuid)/$(duid).arrow"))
    extras = JLD2.load_object("data/treatment_effects/$(group)/$(tuid)/$(duid)_extras.jld2")
    estimate, std_error = Empirics.treatment_effect(tes, extras, Symbol(col))
    return (; Dict(
        Symbol("te_$(col)") => estimate,
        Symbol("te_$(col)_se") => std_error,
    )...)
end

function corsample(df, col)
    estimate = corspearman(df.change_market_share, df[!, col])
    boot = map(1:1000) do _
        i = sample(1:nrow(df), nrow(df))
        corspearman(df.change_market_share[i], df[i, col])
    end
    return estimate, std(boot)
end

with_logging() do
    plot_df = DataFrame(Arrow.Table(snakemake.input["group_market_share_changes"]))
    col = snakemake.wildcards["col"]
    transform!(
        plot_df,
        [:transaction_uid, :dealer_uid, :group] => ByRow(extract_tes$col) => AsTable,
    )
    scatter(
        plot_df.change_market_share,
        plot_df[!, "te_$col"],
        yerr = 1.96plot_df[!, "te_$(col)_se"],
        label = nothing,
        alpha = 0.5,
        size = (800, 600),
        xlabel = "Change in Market Share",
        ylabel = "Change in $(replace(col, '_' => ' ') |> titlecase)",
        left_margin = 5Plots.mm,
    )
    hline!([0.0], color = "black", label = nothing)
    vline!([0.0], color = "black", label = nothing)
    correlation, corr_stderr = corsample(plot_df, "te_$col")
    xanchor = quantile(plot_df.change_market_share, 0.01)
    yanchor = quantile(plot_df[!, "te_$col"] .+ 1.96plot_df[!, "te_$(col)_se"], 0.99)
    annotate!(
        [xanchor],
        [yanchor],
        [
            text(
                "Rank correlation: $(round(correlation, digits=4)) ($(round(corr_stderr, digits=4)))",
                8,
                :black,
                :top,
                :left,
            ),
        ],
    )

    x_grid = range(extrema(plot_df.change_market_share)..., length = 500)
    y_grid = predict(loess(plot_df.change_market_share, plot_df[!, "te_$col"]), x_grid)
    plot!(x_grid, y_grid, msc = 2, label = nothing, linestrokewidth = 4)

    png(snakemake.output[1])
end
