using DealerMnA

!isempty(ARGS) && snakemake!(ARGS[1])

with_logging() do
    r = Registrations.registrations()
    B = 1 + div(nrow(r), 10_000)
    open(Arrow.Writer, snakemake.output[1]) do w
        for (i, batch) in enumerate(Iterators.partition(r, 10_000))
            @info "Writing batch $i of $B"
            Arrow.write(w, Registrations.normalize_price(batch))
        end
    end
end
