module Demand

import ..Registrations
import ..Dealers
import ..Geometry: distance, miles

import ..TRANSITION_GID, ..SINGLE_OPERATOR_GID, ..pluck, ..read_arrow, ..snakemake

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
using TensorOperations:
    tensorcontract, tensorcontract!, tensorproduct, @tensor, contract_indices,
    DefaultBackend
using Bumper
import VectorInterface: add!, inner
using Folds
using FoldsThreads
using Graphs
using Combinatorics
import ArchGDAL as AG
import FiniteDiff: finite_difference_gradient, GradientCache

# include("keyedsparsearray.jl")
# __init__() = _generate_method_overrides()

const PRODUCT_ATTRIBUTES = [:is_coupe, :is_suv, :is_pickup, :is_sedan]
const DEMOGRAPHIC_ATTRIBUTES = [:median_earnings_inv, :urban_pct, :black_pct, :hispanic_pct]

ProductInfo = @NamedTuple begin
    is_coupe::Bool
    is_suv::Bool
    is_pickup::Bool
    is_sedan::Bool
    p::Float32
    p_hausman::Float32
    n::Int16
    g::UUID
end

ProductKey = Tuple{String, String, Int64, Float32} # (make, model, model_year, trim_rank)

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
    t::Lookup{3, Tuple{Int64, Int64}} # (year, quarter)
    j::Lookup{3, ProductKey} # (make, model, model_year, trim_rank)
    f::Lookup{3, UUID}       # dealer uid
    make::Lookup{3, String}  # make
    tjf::Lookup{3, Tuple{Tuple{Int64, Int64}, ProductKey, UUID}} # (t, j, f)
end

function _ProductLookup(kwargs)
    return ProductLookup(
        merge(
            kwargs,
            (;
                make = Lookup{3, String}(),
                tjf = Lookup{3, Tuple{Tuple{Int64, Int64}, ProductKey, UUID}}()),
        ),
    )
end

CountsLookup = @NamedTuple begin
    z::Lookup{4, String}               # zip code
    t::Lookup{4, Tuple{Int64, Int64}} # (year, quarter)
    j::Lookup{4, ProductKey} # (make, model, model_year, trim_rank)
    f::Lookup{4, UUID} # dealer uid
end

OwnershipLookup = @NamedTuple begin
    t::Lookup{3, Tuple{Int64, Int64}} # (year, quarter)
    f::Lookup{3, UUID}                # dealer uid
    g::Lookup{3, UUID}                # group id
end

AdjacencyLookup = @NamedTuple begin
    t::Lookup{4, Tuple{Int64, Int64}} # (year, quarter)
    j::Lookup{4, ProductKey} # (make, model, model_year, trim_rank)
    f::Lookup{4, UUID}                # dealer uid
    g::Lookup{4, UUID}                # group id
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
