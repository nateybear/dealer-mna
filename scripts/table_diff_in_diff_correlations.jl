using DealerMnA
using StatsPlots
using Plots
using NamedArrays
using HypothesisTests
using Combinatorics
using LaTeXStrings
using LaTeXTabulars

snakemake!()

titleize(x) = titlecase(replace(string(x), "_" => " "))

with_logging() do
    tes = JLD2.load_object(snakemake.input["treatment_effect"])
    statistics = collect(keys(tes))
    N = length(statistics)
    out = NamedArray(repeat([L"-"], N, N), (statistics, statistics))
    for (a, b) in combinations(statistics, 2)
        combined = innerjoin(tes[a].extras, tes[b].extras, on = :dealer_uid, makeunique = true)
        test = CorrelationTest(combined.diff_in_diff, combined.diff_in_diff_1)
        significant = pvalue(test) < 0.05 ? "^{***}" : ""
        statistic = round(test.r, digits = 3)
        out[a, b] = LaTeXString("\$$(statistic)$(significant)\$")
    end
    tablenames = titleize.(names(out, 1))
    open(snakemake.output[1], "w") do io
        latex_tabular(
            io,
            Tabular("r|" * "c"^(Base.size(out, 1) - 1)),
            [
                Rule(:top),
                ["", tablenames[2:end]...],
                map(splat(vcat), zip(tablenames[1:(end - 1)], eachrow(out[1:(end - 1), 2:end])))...,
                Rule(:bottom),
            ],
        )
    end
end
