using DealerMnA

!isempty(ARGS) && snakemake!(ARGS[1])

with_logging() do
    open(snakemake.output[1], "w") do io
        foreach(println$io, unique(Registrations.price_table().make))
    end
end
