module Registrations

export registrations, add_zips, normalize_body_class, normalize_price

using DrWatson
using DataFrames
using Arrow
using CSV
using Memoize
using PartialFunctions
using Lazy: @>

@memoize registrations() = DataFrame(Arrow.Table(datadir("arrow/new_registrations.arrow")))

include("zips.jl")
include("normalize_price.jl")
include("normalize_body_class.jl")

end
