using DealerMnA

function (@main)(_)
    with_logging() do
        Dealers.init!()
        Dealers.write_to_disk!(snakemake.output[1])
    end
    return nothing
end
