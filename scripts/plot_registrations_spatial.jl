using DealerMnA
using Plots

snakemake!()

with_logging() do
    zips = Registrations.zips()
    zip_counts = @> snakemake.input["new_registrations"] begin
        Arrow.Table()
        DataFrame()
        select(:ownrzpcd => :zip_code)
        groupby(:zip_code)
        combine(nrow)
    end
    leftjoin!(zips, zip_counts, on = :zip_code)
    zips.nrow = coalesce.(zips.nrow, 0)
    sort!(zips, :nrow)
    scatter(
        first.(zips.geometry),
        last.(zips.geometry),
        marker_z = zips.nrow,
        seriescolor = cgrad(:algae),
        alpha = 0.75,
        markerstrokewidth = 0,
        xaxis = nothing,
        yaxis = nothing,
        label = "Registrations/Zip Code",
        size = (800, 600),
    )
    png(snakemake.output[1])
end
