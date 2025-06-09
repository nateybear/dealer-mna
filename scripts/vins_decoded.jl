using DealerMnA

!isempty(ARGS) && snakemake!(ARGS[1])

with_logging() do
    in_writer = CSV.Chunks(snakemake.input[1], stringtype = String)
    open(Arrow.Writer, snakemake.output[1]) do out_writer
        for chunk in in_writer
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
        end
    end
end
