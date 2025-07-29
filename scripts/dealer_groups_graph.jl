#### IMPORTANT: DON'T RUN THIS VIA SNAKEMAKE, IT NEEDS TO BE RUN MANUALLY
using DealerMnA
using Base.Iterators: flatten

function (@main)(_)
    Dealers.load_from_disk!("data/dealers_init.jld2")
    all_entities =
        Dealers.State.df.entity ∪
        reduce(union, (Dealers.entities(e) for e in Dealers.Events))
    g = Dealers.new_dealer_graph()
    for entity in all_entities
        if !haskey(g, entity)
            try
                Dealers.search!(g, entity)
            catch err
                @warn "Failed to search for entity '$entity', assuming it is a person" err
                g[entity] = Dealers.PersonNode(entity)
            end
            sleep(0.1)  # to avoid hitting the API too hard
        end
    end
    Dealers.color!(g)
    JLD2.save_object("data/dealer_groups_graph.jld2", g)
    return nothing
end
