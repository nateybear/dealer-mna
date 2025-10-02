module Geometry

export gdf, distance, catchment_area, with_map

import ArchGDAL as AG
import GeoDataFrames as GDF
import GeoFormatTypes as GFT

using DelaunayTriangulation
using LinearAlgebra
using Memoize

using DataFrames,
    Lazy,
    PartialFunctions,
    Missings,
    Memoize,
    Combinatorics,
    DrWatson

include("crs.jl")
include("conversions.jl")
include("catchment_areas.jl")
include("mapping.jl")

using Unitful
using Unitful.DefaultSymbols
const mi = u"mi"

### UNSAFE: distance assumes correct CRS for performance
distance(a::AG.IGeometry, b::AG.IGeometry) = AG.distance(a, b)m # distance between two geometries in meters
distance(a::AG.IGeometry, b) = distance(a, AG.createpoint(b)) # distance between a geometry and a point
distance(a, b) = norm(a .- b)m # distance between two points

miles(d) = ustrip(uconvert(mi, d)) # convert distance to miles
miles(a, b) = miles(distance(a, b)) # distance between two geometries in miles

#### reading function that asserts the correct CRS
function gdf(filename, args...; kwargs...)
    out = GDF.read(filename, args...; kwargs...)
    foreach(assert_crs!, out.geometry)
    return out
end

end
