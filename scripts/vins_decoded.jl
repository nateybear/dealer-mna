using DealerMnA

snakemake!()

with_logging() do
    in_writer = CSV.Chunks(
        snakemake.input[1],
        ntasks = 1000, # about 80,000 rows per chunk
        stringtype = String, 
        quotechar = '"',
    )
    open(Arrow.Writer, snakemake.output[1]) do out_writer
        for (i, chunk) in enumerate(in_writer)
            # Clean the chunk
            cleaned_chunk = @> chunk begin
                DataFrame
                polish_names_ROT
                select(
                    :vin,
                    :make,
                    :model,
                    :model_year,
                    :series,
                    :trim,
                    :body_class,
                    r"^plant",
                    r"^fuel_type",
                    r"^displacement",
                )
            end

            # Write the cleaned chunk to the output
            Arrow.write(out_writer, cleaned_chunk)

            if i % 5 == 0
                @info "Processed chunk $i"
            end
        end
    end
end
