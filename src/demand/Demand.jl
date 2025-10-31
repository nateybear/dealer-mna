module Demand

import ..Registrations
import ..Dealers
import ..Geometry: distance, miles

import ..TRANSITION_GID, ..SINGLE_OPERATOR_GID, ..pluck

using UUIDs
using PartialFunctions
using DataFrames
using Lazy: @>
using Random
using LogExpFunctions: softmax!
using Distributions
using OnlineStats
using NamedArrays
import Base.Iterators: partition
using Dates
using DataStructures: DefaultOrderedDict # maintain insertion order
using Memoize
using LinearAlgebra
using StatsBase
using Base.Threads: @threads
using SparseArrayKit
using TensorOperations: tensorcontract, tensorproduct
using Folds
using FoldsThreads

const PRODUCT_ATTRIBUTES = [:domestic]
const DEMOGRAPHIC_ATTRIBUTES = [:median_earnings_inv, :urban_pct, :black_pct, :hispanic_pct]

const DOMESTIC_MAKES = Set([
    "CHEVROLET",
    "CHRYSLER",
    "DODGE",
    "FORD",
    "GMC",
    "JEEP",
    "LINCOLN",
    "RAM",
])

ProductInfo = @NamedTuple begin
    domestic::Bool
    p::Float32
    n::Int16
    g::UUID
end

AgentInfo = NamedTuple{
    tuple(DEMOGRAPHIC_ATTRIBUTES..., :M),
    NTuple{length(DEMOGRAPHIC_ATTRIBUTES) + 1, Float32},
}

# for a given key, return the list of Cartesian indices in the sparse array
# where that key appears
Lookup{N, T} = DefaultOrderedDict{T, Vector{CartesianIndex{N}}}
function Lookup{N, T}() where {N, T}
    return DefaultOrderedDict{T, Vector{CartesianIndex{N}}}(Vector{CartesianIndex{N}})
end

ProductLookup = @NamedTuple begin
    t::Lookup{4, Tuple{Int64, Int64}} # (year, quarter)
    f::Lookup{4, UUID}                # dealer uid
    j::Lookup{4, Tuple{String, String, Int64}} # (make, model, model_year)
    k::Lookup{4, Float32}             # trim rank
    make::Lookup{4, String}           # make
    tjf::Lookup{4, Tuple{Tuple{Int64, Int64}, UUID, Tuple{String, String, Int64}}} # (t, f, j)
end
function new_product_lookup(df)
    out = ProductLookup((;
        t = Lookup{4, Tuple{Int64, Int64}}(),
        j = Lookup{4, Tuple{String, String, Int64}}(),
        f = Lookup{4, UUID}(),
        k = Lookup{4, Float32}(),
        make = Lookup{4, String}(),
        tjf = Lookup{4, Tuple{Tuple{Int64, Int64}, UUID, Tuple{String, String, Int64}}}(),
    ))
    for idx in (:t, :j, :k, :f)
        for val in sort(unique(df[!, idx]))
            empty!(out[idx][val]) # force key creation to preserve order
        end
    end
    return out
end

CountsLookup = @NamedTuple begin
    t::Lookup{5, Tuple{Int64, Int64}} # (year, quarter)
    f::Lookup{5, UUID} # dealer uid
    j::Lookup{5, Tuple{String, String, Int64}} # (make, model, model_year)
    k::Lookup{5, Float32} # trim rank
    z::Lookup{5, String}               # zip code
end
function new_counts_lookup(df)
    out = CountsLookup((;
        t = Lookup{5, Tuple{Int64, Int64}}(),
        f = Lookup{5, UUID}(),
        j = Lookup{5, Tuple{String, String, Int64}}(),
        k = Lookup{5, Float32}(),
        z = Lookup{5, String}(),
    ))
    for idx in (:t, :j, :k, :f, :z)
        for val in sort(unique(df[!, idx]))
            empty!(out[idx][val]) # force key creation to preserve order
        end
    end
    return out
end

# so this works with sparse arrays
Base.iszero(::NamedTuple) = false
Base.zero(::Type{<:NamedTuple}) = NamedTuple()

include("loading.jl")
include("models.jl")
include("estimation.jl")

end
