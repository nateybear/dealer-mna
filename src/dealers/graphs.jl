using Graphs
using MetaGraphsNext
using UUIDs
using Lazy: @>
using DataFrames

import JSON3
import HTTP
using ..Limiter
import DotEnv
DotEnv.load!(ENV, here("src/dealers/.env"), override = true)
const api_token = get(ENV, "OPENCORPORATES_API_TOKEN", nothing)
using Unitful.DefaultSymbols: s
using Unitful: @u_str
const limiter = RateLimiter(2 / s, 10_000 / u"d")

abstract type Node end

mutable struct DealerNode <: Node
    name::String
    id::Union{UUID, Nothing}
    DealerNode(name) = new(name, nothing)
end

mutable struct PersonNode <: Node
    name::String
    id::Union{UUID, Nothing}
    PersonNode(name) = new(name, nothing)
end

new_dealer_graph() = MetaGraph(
    Graph(),
    String, # label type
    Node, # vertex type
    Vector{String}, # edge type
)

const DealerGraph = typeof(new_dealer_graph())

function search!(g::DealerGraph, name::String, active = false)
    isnothing(api_token) &&
        error("No OpenCorporates API token found in environment variables.")
    resp = HTTP.get(
        limiter,
        "https://api.opencorporates.com/companies/search",
        query = Dict(
            "q" => name,
            "api_token" => api_token,
            "jurisdiction_code" => "us_tx",
            "order" => "score",
            "normalise_company_name" => "true",
            (active ? ["inactive" => "false"] : [])...,
        ),
    )
    results = JSON3.read(resp.body, Dict{String, Any})["results"]["companies"]
    isempty(results) && error("No results found for '$name' in Texas.")
    company_id = first(results)["company"]["company_number"]
    if isnothing(company_id)
        @warn "No company ID found for '$name'. Assuming it's a person."
        g[name] = PersonNode(name)
        return nothing
    end
    return insert_entity!(g; name, company_id)
end

function insert_entity!(
    g::DealerGraph;
    name::String,
    company_id::String,
    jurisdiction_code = "us_tx",
)
    g[name] = DealerNode(name)
    resp = try
        HTTP.get(
            limiter,
            "https://api.opencorporates.com/companies/$jurisdiction_code/$company_id",
            query = Dict("api_token" => api_token),
        )
    catch err
        @warn "Failed to fetch company data for $company_id in jurisdiction $jurisdiction_code" err
        return nothing
    end
    company_data = JSON3.read(resp.body, Dict{String, Any})["results"]["company"]

    # check board members
    board = company_data["officers"]
    if !isnothing(board) && length(board) > 0
        board = DataFrame(
            map(board) do officer
                officer = first(values(officer))
                (;
                    name = "$(officer["name"]) ($(officer["id"]))",
                    position = something(officer["position"], "nothing"),
                )
            end,
        )
        subset!(board, :position => ByRow(!=("agent")))
        insert_board!(g, (name, board))
    end

    # check if we have a controlling entity
    controlling_entity = company_data["controlling_entity"]
    if !isnothing(controlling_entity)
        # it's a placeholder if OC is unable to obtain info on the entity
        if !contains(controlling_entity["opencorporates_url"], "placeholder")
            if !haskey(g, controlling_entity["name"])
                insert_entity!(
                    g,
                    name = controlling_entity["name"],
                    company_id = controlling_entity["company_number"],
                    jurisdiction_code = controlling_entity["jurisdiction_code"],
                )
            end
        else
            @warn "Controlling entity '$(controlling_entity["name"])' for '$name' is a placeholder"
            g[controlling_entity["name"]] = DealerNode(controlling_entity["name"])
        end
        g[name, controlling_entity["name"]] = ["controlling entity"]
    end

    # check if we have a company name that we can link to
    company_name = get(company_data, "company_name", nothing)
    if !isnothing(company_name) && company_name != name
        if !haskey(g, company_name)
            insert_entity!(g; name = company_name, company_id, jurisdiction_code)
        end
        g[name, company_name] = ["also known as"]
    end
    alternative_names = get(company_data, "alternative_names", nothing)
    if !isnothing(alternative_names)
        for alt_name in alternative_names
            if !haskey(g, alt_name)
                insert_entity!(g; name = alt_name, company_id, jurisdiction_code)
            end
            g[name, alt_name] = ["also known as"]
        end
    end
    return g
end

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
    return delete!(g, from_label)
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

function color!(g, node, uuid = uuid4())
    node.id = uuid
    for neighbor in neighbor_labels(g, node.name)
        neighbor_node = g[neighbor]
        isnothing(neighbor_node.id) && color!(g, neighbor_node, uuid)
    end
end

color!(g) =
    while true
        node = next_uncolored(g)
        isnothing(node) && break
        color!(g, node)
    end

reset!(g) =
    for l in labels(g)
        g[l].id = nothing
    end
