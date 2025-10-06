using DealerMnA
using Dates

using DataStructures: counter

!isempty(ARGS) && snakemake!(ARGS[1])

with_logging() do
    Dealers.load_from_disk!(snakemake.input["dealers_init"])
    open(Arrow.Writer, snakemake.output["new_registrations"]) do out_writer
        for chunk in Arrow.Stream(snakemake.input["unmatched_registrations"])
            @info "Processing chunk of registrations"
            registrations = DataFrame(chunk)
            registrations.quarter = Dates.quarter.(registrations.regeffdate)
            registrations.year = Dates.year.(registrations.regeffdate)
            registrations.t = tuple.(registrations.year, registrations.quarter)
            registrations = Registrations.normalize_price(registrations)
            registrations = combine(groupby(registrations, :t)) do r
                t = first(r.t)
                Dealers.fuzzytravel(t) do state
                    r = Dealers.add_distance(r; state)
                    return ismissing(r) ? DataFrame() : r
                end
            end
            Arrow.write(out_writer, registrations)
        end
    end
end
