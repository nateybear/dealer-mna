using DealerMnA

using LaTeXStrings

!isempty(ARGS) && snakemake!(ARGS[1])

process_facts(facts) =
    mapreduce((@$ join(_, "\n")), facts) do (label, fact)
        "\\newcommand{$label}{$fact}"
    end

with_logging() do
    Dealers.load_from_disk!(snakemake.input["dealers_init"])
    facts = Dict{String, Any}()
    registrations = Registrations.normalize_price(Registrations.registrations())
    push!(facts, "numregistrationstotal" => nrow(registrations))

    open(snakemake.output[1], "w") do io
        print(io, process_facts(facts))
    end
end
