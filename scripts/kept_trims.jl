using DealerMnA
using DataFrames
using Lazy: @>
using Arrow
using StatsBase: mean

AVG_Q_THRESHOLD = 3

function (@main)(_)
    snakemake!()
    with_logging() do
        fords = read_arrow(snakemake.input["fords"])

        dealer_quarter_trim = @> fords groupby([:model, :model_year, :extra, :dealer_uid, :t]) combine(
            nrow => :dq_sales
        )
        dq_counts = @> dealer_quarter_trim groupby([:model, :model_year, :extra]) combine(
            :dq_sales => mean
        )

        kept = subset(dq_counts, :dq_sales_mean => ByRow(>=(AVG_Q_THRESHOLD)))
        dropped = subset(dq_counts, :dq_sales_mean => ByRow(<(AVG_Q_THRESHOLD)))
        @info "Kept trims" n_kept = nrow(kept) n_dropped = nrow(dropped) n_total = nrow(dq_counts) threshold =
            AVG_Q_THRESHOLD
        @info "Dropped trims" trims = collect(
            zip(dropped.model, dropped.extra, dropped.dq_sales_mean)
        )

        return Arrow.write(
            snakemake.output["kept_trims"], select(kept, :model, :model_year, :extra)
        )
    end
    return nothing
end
