using DealerMnA
using DataFrames
using Lazy: @>
using Arrow

# Trims kept for the BBL supply pipeline. Cutoff at 3% of Ford dealer-quarters
# — matches the natural shoulder in tables/trim_variation_dealer_quarter.csv.
# The share denominator is the same one that plot_assortment_probs.jl reports,
# so the two stay comparable.
const THRESHOLD_SHARE = 0.03

function (@main)(_)
    snakemake!()
    with_logging() do
        fords = read_arrow(snakemake.input["fords"])

        dealer_quarter_trim = @> fords groupby([:model, :extra, :dealer_uid, :t]) combine(
            nrow => :_sales
        )
        dq_counts = @> dealer_quarter_trim groupby([:model, :extra]) combine(
            nrow => :n_dq_with_trim
        )
        n_total_dq = nrow(unique(select(fords, :dealer_uid, :t)))
        dq_counts.share = dq_counts.n_dq_with_trim ./ n_total_dq

        kept = subset(dq_counts, :share => ByRow(>=(THRESHOLD_SHARE)))
        dropped = subset(dq_counts, :share => ByRow(<(THRESHOLD_SHARE)))
        @info "Kept trims" n_kept = nrow(kept) n_dropped = nrow(dropped) n_total = nrow(dq_counts) threshold = THRESHOLD_SHARE
        @info "Dropped trims" trims = collect(zip(dropped.model, dropped.extra, dropped.share))

        Arrow.write(snakemake.output["kept_trims"], select(kept, :model, :extra))
    end
    return nothing
end
