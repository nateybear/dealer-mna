using DealerMnA

using DataStructures: counter

function (@main)(_)
    with_logging() do
        Dealers.load_from_disk!(snakemake.input["dealers_init"])
        current_gdns = reduce(union, Dealers.State.df.gdns)
        event_gdns = mapreduce(Dealers.affected_gdns, union, Dealers.Events)
        all_gdns = union(current_gdns, event_gdns)
        open(Arrow.Writer, snakemake.output["new_registrations"]) do out_writer
            for chunk in Arrow.Stream(snakemake.input["unmatched_registrations"])
                @info "Processing chunk of registrations"
                registrations = @> chunk begin
                    DataFrame
                    subset(:dlrgdn => ByRow(in(all_gdns)))
                end
                Arrow.write(out_writer, registrations)
            end
        end
    end
    return nothing
end
