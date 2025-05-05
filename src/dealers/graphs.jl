using Graphs
using MetaGraphsNext
using UUIDs
using Lazy: @>
using DataFrames

abstract type Node end

mutable struct DealerNode <: Node
    name::String
    id::Union{UUID,Nothing}
    DealerNode(name) = new(name, nothing)
end

mutable struct PersonNode <: Node
    name::String
    id::Union{UUID,Nothing}
    PersonNode(name) = new(name, nothing)
end

new_dealer_graph() = MetaGraph(
    Graph(),
    String, # label type
    Node, # vertex type
    Vector{String} # edge type
)

const DealerGraph = typeof(new_dealer_graph())

function insert_board!(g, (dealer, board))
    g[dealer] = DealerNode(dealer)
    if !isnothing(board) && ncol(board) > 0
        grouped_board = @> board groupby(:name) pairs
        for ((; name), (; position)) in grouped_board
            name = titlecase(name)
            if !haskey(g, name)
                g[name] = PersonNode(name)
            end
            g[name, dealer] = position
        end
    end
end

function move!(g, from_label, to_label)
    if !haskey(g, to_label)
        g[to_label] = PersonNode(to_label)
    end
    for neighbor in neighbor_labels(g, from_label)
        if haskey(g, neighbor, to_label)
            union!(g[neighbor, to_label], g[from_label, neighbor])
        else
            g[neighbor, to_label] = g[from_label, neighbor]
        end
    end
    delete!(g, from_label)
end

function next_uncolored(g)
    for l in labels(g)
        v = g[l]
        if isnothing(v.id)
            return v
        end
    end
    return nothing
end

function color!(g, node, uuid=uuid4())
    node.id = uuid
    for neighbor in neighbor_labels(g, node.name)
        neighbor_node = g[neighbor]
        isnothing(neighbor_node.id) && color!(g, neighbor_node, uuid)
    end
end

function color!(g)
    while true
        node = next_uncolored(g)
        isnothing(node) && break
        color!(g, node)
    end
end

function reset!(g)
    for l in labels(g)
        g[l].id = nothing
    end
end
