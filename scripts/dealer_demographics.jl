using DealerMnA

using Unitful
using Unitful.DefaultSymbols

to_miles = ustrip ∘ uconvert$(u"mi")

function (@main)(_)
    with_logging() do
        Dealers.load_from_disk!(snakemake.input["dealers_init"])
        intersects = !isempty ∘ intersect
        out = mapreduce(vcat, eachrow(Dealers.State.df)) do d
            relevant_dealers = subset(
                Dealers.State.df,
                :franchises => ByRow(intersects$d.franchises),
            )
            distances =
                to_miles.(Geometry.distance.(relevant_dealers.geometry, Ref(d.geometry)))
            order = sortperm(distances)[2:4]
            DataFrame(
                uid = d.uid,
                total_population = Empirics.total_population(d),
                median_income = Empirics.median_income(d),
                share_urban = Empirics.share_urban(d),
                share_white = Empirics.share_white(d),
                share_black = Empirics.share_black(d),
                share_hispanic = Empirics.share_hispanic(d),
                closest_first = distances[order[1]],
                closest_second = distances[order[2]],
                closest_third = distances[order[3]],
            )
        end
        Arrow.write(
            snakemake.output[1],
            out,
        )
    end
    return nothing
end
