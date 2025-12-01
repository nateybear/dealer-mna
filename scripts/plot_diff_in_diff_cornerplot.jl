using DealerMnA
using StatsPlots
using Plots

!isempty(ARGS) && snakemake!(ARGS[1])

with_logging() do
    tes = JLD2.load_object(snakemake.input["treatment_effect"])
    plot_df = mapreduce(innerjoin$(on=:dealer_uid,), pairs(tes)) do (stat, result)
        DataFrames.rename(result.extra, :diff_in_diff => stat)
    end

    titleize(x) = titlecase(replace(string(x), "_" => " "))
    DataFrames.rename!(titleize, plot_df)

    p = @df plot_df cornerplot(
        cols(4:9), 
        size=(1400, 900), 
        compact=true, 
        left_margin=10Plots.mm, 
        bottom_margin=10Plots.mm
    )

    # fix the trim_rank plots
    # for idx in 15:19
    #     ylims!(p[idx], (-0.15, 0.15))
    # end

    png(p, snakemake.output[1])
end
