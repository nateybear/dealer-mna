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
using Base.Threads: @threads, @spawn
using SparseArrayKit
using TensorOperations: tensorcontract, tensorproduct, @tensor
using Folds
using FoldsThreads
using Graphs
using Combinatorics
import ArchGDAL as AG
import FiniteDiff: finite_difference_gradient, GradientCache

# include("keyedsparsearray.jl")
# __init__() = _generate_method_overrides()

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
    p_hausman::Float32
    n::Int16
    g::UUID
end

AgentInfo = NamedTuple{
    tuple(DEMOGRAPHIC_ATTRIBUTES..., :M),
    NTuple{length(DEMOGRAPHIC_ATTRIBUTES) + 1, Float32},
}

# for a given key, return the list of Cartesian indices in the sparse array
# where that key appears
Lookup{N, T} = DefaultOrderedDict{T, Set{CartesianIndex{N}}}
Lookup{N, T}() where {N, T} = DefaultOrderedDict{T, Set{CartesianIndex{N}}}(Set{CartesianIndex{N}})

ProductLookup = @NamedTuple begin
    t::Lookup{4, Tuple{Int64, Int64}} # (year, quarter)
    j::Lookup{4, Tuple{String, String, Int64}} # (make, model, model_year)
    k::Lookup{4, Float32}             # trim rank
    f::Lookup{4, UUID}                # dealer uid
    make::Lookup{4, String}           # make
    tjf::Lookup{4, Tuple{Tuple{Int64, Int64}, Tuple{String, String, Int64}, UUID}} # (t, j, f)
end

function _ProductLookup(kwargs)
    return ProductLookup(
        merge(
            kwargs,
            (;
                make = Lookup{4, String}(),
                tjf = Lookup{4, Tuple{Tuple{Int64, Int64}, Tuple{String, String, Int64}, UUID}}()),
        ),
    )
end

CountsLookup = @NamedTuple begin
    z::Lookup{5, String}               # zip code
    t::Lookup{5, Tuple{Int64, Int64}} # (year, quarter)
    j::Lookup{5, Tuple{String, String, Int64}} # (make, model, model_year)
    k::Lookup{5, Float32} # trim rank
    f::Lookup{5, UUID} # dealer uid
end

OwnershipLookup = @NamedTuple begin
    t::Lookup{3, Tuple{Int64, Int64}} # (year, quarter)
    f::Lookup{3, UUID}                # dealer uid
    g::Lookup{3, UUID}                # group id
end

AdjacencyLookup = @NamedTuple begin
    t::Lookup{4, Tuple{Int64, Int64}} # (year, quarter)
    j::Lookup{4, Tuple{String, String, Int64}} # (make, model, model_year)
    f::Lookup{4, UUID}                # dealer uid
    g::Lookup{4, UUID}                # group id
end

TrimFreeLookup = @NamedTuple begin
    t::Lookup{3, Tuple{Int64, Int64}}
    j::Lookup{3, Tuple{String, String, Int64}}
    f::Lookup{3, UUID}
end

function new_lookup(T, df::DataFrame, cols...)
    out = T((; (col => Lookup{length(cols), eltype(df[!, col])}() for col in cols)...))
    for idx in cols
        for val in sort(unique(df[!, idx]))
            empty!(out[idx][val]) # force key creation to preserve order
        end
    end
    return out
end
function new_lookup(T; kwargs...)
    k_max, v_max = argmax(pairs(kwargs)) do (_, v)
        length(v)
    end
    df = insertcols(
        DataFrame(k_max => v_max),
        (k => first(Iterators.cycle(v), length(v_max)) for (k, v) in pairs(kwargs) if k ≠ k_max)...,
    )
    return new_lookup(T, df, keys(kwargs)...)
end

# so this works with sparse arrays
Base.iszero(::NamedTuple) = false
Base.zero(::Type{<:NamedTuple}) = NamedTuple()

include("loading.jl")
include("models.jl")
include("estimation.jl")

end
