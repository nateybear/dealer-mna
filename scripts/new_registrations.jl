using DealerMnA
using Dates

using DataStructures: counter

snakemake!()

with_logging() do
    Dealers.load_from_disk!(snakemake.input["dealers_init"])
    open(Arrow.Writer, snakemake.output["new_registrations"]) do out_writer
        for chunk in Arrow.Stream(snakemake.input["unmatched_registrations"])
            @info "Processing chunk of registrations"
            registrations = DataFrame(chunk)
            registrations.t = firstdayofquarter.(registrations.regeffdate)
            registrations = Registrations.normalize_price(registrations)
            registrations = Registrations.add_manufacturer(registrations)
            isempty(registrations) && continue
            registrations = combine(groupby(registrations, :t)) do r
                t = first(r.t)
                return Dealers.fuzzytravel(t) do
                    r = Dealers.add_distance(r)
                    return ismissing(r) ? DataFrame() : r
                end
            end
            Arrow.write(out_writer, registrations)
        end
    end
end
