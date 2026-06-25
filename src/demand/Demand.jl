module Demand

import ..Registrations
import ..Dealers
import ..Geometry: distance, miles

import ..TRANSITION_GID, ..SINGLE_OPERATOR_GID, ..pluck, ..read_arrow, ..snakemake

using UUIDs
using PartialFunctions
using DataFrames
using Lazy: @>
using OnlineStats
using NamedArrays
using Dates
using DataStructures: OrderedDict
using Memoize
using LinearAlgebra
using SparseArrays
using StatsBase
using Base.Threads: @threads
using Folds
using FoldsThreads
using Graphs
using Combinatorics
import ArchGDAL as AG
import JLD2
import StatsAPI: predict, coef, coefnames

const PRODUCT_ATTRIBUTES = [:is_coupe, :is_suv, :is_pickup, :is_sedan]
const DEMOGRAPHIC_ATTRIBUTES = [:median_earnings_inv, :urban_pct, :black_pct, :hispanic_pct]

const QuarterKey = Tuple{Int64, Int64}                          # (year, quarter)
const VehicleKey = Tuple{String, String, Int64, Float32}        # (make, model, model_year, trim_rank)
const TJFKey = Tuple{QuarterKey, VehicleKey, UUID}              # (t, j, f)

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

AgentInfo = NamedTuple{
    tuple(DEMOGRAPHIC_ATTRIBUTES..., :M),
    NTuple{length(DEMOGRAPHIC_ATTRIBUTES) + 1, Float32},
}

# Flat-index lookups. `t`/`j`/`f`/`make` are one-to-many (multiple products share a key);
# `tjf` is one-to-one (each (t,j,f) triple maps to a unique product δ index).
struct ProductLookup
    t::OrderedDict{QuarterKey, Vector{Int32}}
    j::OrderedDict{VehicleKey, Vector{Int32}}
    f::OrderedDict{UUID, Vector{Int32}}
    make::OrderedDict{String, Vector{Int32}}
    tjf::OrderedDict{TJFKey, Int32}
end

include("models.jl")
include("loading.jl")
include("dense.jl")

end
