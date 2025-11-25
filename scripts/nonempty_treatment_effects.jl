using DealerMnA
import Arrow
using DataFrames
import CSV
using UUIDs

!isempty(ARGS) && snakemake!(ARGS[1])

with_logging() do
    out = DataFrame(transaction_uid = String[], dealer_uid = String[])
    for i in snakemake.input
        @info "Processing $(i)"
        df = read_arrow(i)
        if !isempty(df)
            path, fname = splitdir(i)
            dealer_uid = replace(fname, r"\.arrow$" => "")
            transaction_uid = basename(path)
            push!(out, (; transaction_uid, dealer_uid))
        else
            @warn "Skipping empty dataframe $(i)"
        end
    end
    CSV.write(snakemake.output[1], out)
end
