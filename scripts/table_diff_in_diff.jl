using DealerMnA

using LinearAlgebra
using LaTeXTabulars
using LaTeXStrings
using DataStructures: DefaultDict
using Printf

!isempty(ARGS) && snakemake!(ARGS[1])

function fit_plus_std(x, y; digits = 3)
    significant = abs(x) > 1.96y
    formatstring = "%.$(digits)g"
    x = @eval @sprintf($formatstring, $x)
    y = @eval @sprintf($formatstring, $y)
    return LaTeXString(
        "\\makecell{$x$(significant ? "\\textsuperscript{***}" : "") \\\\ ($y)\\\\[0.25em]}"
    )
end

titleize(s) = @> s string replace("_" => " ") titlecase

function spec_to_dataframe(dict, specifications)
    specifications = collect(pairs(specifications))
    outcomes = sort(collect(keys(first(values(dict)))))
    out = DataFrame(;
        Outcome = titleize.(outcomes),
        (Symbol.(last.(specifications)) .=> Ref(repeat([L""], length(outcomes))))...,
    )
    for (i, outcome) in enumerate(outcomes)
        for (spec_key, spec_name) in specifications
            out[i, spec_name] = fit_plus_std(
                dict[spec_key][outcome].did_estimate,
                dict[spec_key][outcome].did_stderr,
            )
        end
    end
    return out
end

with_logging() do
    dict = DefaultDict{String, Dict}(Dict{String, Any})
    specifications = Dict(
        "twfe" => "TWFE",
        "sunabraham" => "Sun-Abraham",
        "synthetic" => "Synthetic Control",
    )
    directions = ["increasing", "decreasing"]
    for (direction, specification) in Iterators.product(directions, keys(specifications))
        filename = filter(snakemake.input["treatment_effects"]) do f
            occursin(direction, f) && occursin(specification, f)
        end
        filename = only(filename)
        dict[direction][specification] = JLD2.load_object(filename)
    end

    increasing_table = spec_to_dataframe(
        dict["increasing"],
        specifications,
    )
    decreasing_table = spec_to_dataframe(
        dict["decreasing"],
        specifications,
    )

    open(snakemake.output[1], "w") do io
        latex_tabular(
            io,
            Tabular("l|" * "c"^length(specifications)),
            [
                Rule(:top),
                names(increasing_table),
                Rule(:mid),
                ["Increasing Market Share"],
                Rule(:mid),
                collect.(eachrow(increasing_table))...,
                Rule(:mid),
                ["Decreasing Market Share"],
                Rule(:mid),
                collect.(eachrow(decreasing_table))...,
                Rule(:bottom),
            ],
        )
    end
end
