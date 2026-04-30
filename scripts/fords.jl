using DealerMnA

function (@main)(_)
    snakemake!()
    with_logging() do
        Dealers.load_from_disk!(snakemake.input["dealers_init"])
        registrations = read_arrow(snakemake.input["new_registrations"])
        ford_dealers = subset(Dealers.State.df, :franchises => ByRow(∋("FORD"))).uid
        fords = subset(
            registrations,
            :dealer_uid => ByRow(∈(ford_dealers)),
            :make => ByRow(==("FORD")),
            view = true,
        )
        fords = let appropriate_t = (m, t) -> (Date(m) - Quarter(1)) <= t <= (Date(m) + Year(1))
            subset(fords, [:model_year, :t] => ByRow(appropriate_t), view = true)
        end
        Arrow.write(snakemake.output[1], fords)
    end
    return nothing
end
