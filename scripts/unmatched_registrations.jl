using DealerMnA

using DataStructures: counter

function (@main)(_)
    with_logging() do
        open(Arrow.Writer, snakemake.output["unmatched_registrations"]) do out_writer
            for chunk in Arrow.Stream(snakemake.input["registrations"])
                @info "Processing chunk of registrations"
                registrations = DataFrame(chunk)
                registrations = dropmissing(registrations, [:dlrgdn, :make, :model])
                registrations = @> registrations begin
                    subset(
                        :vehodmtrreading => ByRow(<(250)),
                        :vehsalesprice => ByRow(>(0)),
                        [:regeffdate, :model_year] =>
                            ByRow((r, m) -> abs(year(r) - m) <= 1),
                        skipmissing = true,
                    )
                    subset(:ttlownrtypecd => ByRow(x -> ismissing(x) || x in ("IN", "EI")))
                    Registrations.filter_zips
                    Registrations.normalize_body_class
                end
                Arrow.write(out_writer, registrations)
            end
        end

        # compile popular makes and filter registrations based on them
        make_counter = counter(Tuple{String, String})
        for chunk in Arrow.Stream(snakemake.output["unmatched_registrations"])
            merge!(make_counter, counter(zip(chunk.make, chunk.model)))
        end
        make_counter = filter(x -> x.second >= 10_000, make_counter)
        popular_models = DataFrame(;
            make = first.(keys(make_counter)),
            model = last.(keys(make_counter)),
            count = collect(values(make_counter)),
        )
        subset!(popular_models, :make => ByRow(!=("BUICK"))) # Buick got discontinued, can't reconstruct dealer network
        open(Arrow.Writer, snakemake.output["popular_models"]) do out_writer
            Arrow.write(out_writer, popular_models)
        end
        open(Arrow.Writer, "data/temp.arrow") do out_writer
            for chunk in Arrow.Stream(snakemake.output["unmatched_registrations"])
                @info "Filtering registrations based on popular makes"
                registrations = DataFrame(chunk)
                registrations = semijoin(
                    registrations,
                    popular_models,
                    on = [:make, :model],
                    makeunique = true,
                )
                Arrow.write(out_writer, registrations)
            end
        end
        mv("data/temp.arrow", snakemake.output["unmatched_registrations"], force = true)
    end
    return nothing
end
