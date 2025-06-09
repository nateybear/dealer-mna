using DealerMnA

import DataFrames: unstack
using StatsBase: mode
using LaTeXTabulars

!isempty(ARGS) && snakemake!(ARGS[1])

with_logging() do
    table = @> Registrations.registrations() begin
        groupby([:make, :model])
        combine(:body_class => mode, renamecols = false)
        groupby([:make, :body_class])
        combine(nrow)
        unstack(:body_class, :nrow, fill = 0)
        sort(:make)
    end

    body_classes = names(table, Not(:make))

    open(snakemake.output[1], "w") do io
        latex_tabular(
            io,
            Tabular("l" * "c"^length(body_classes)),
            [
                Rule(:top),
                ["", body_classes...],
                Rule(:mid),
                collect.(eachrow(table))...,
                Rule(:bottom),
            ],
        )
    end
end
