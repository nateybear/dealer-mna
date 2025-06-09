using DealerMnA

!isempty(ARGS) && snakemake!(ARGS[1])

with_logging() do
    open(snakemake.output[1], "w") do io
        for s in keys(Empirics.STATISTICS)
            println(io, s)
        end
    end
end
