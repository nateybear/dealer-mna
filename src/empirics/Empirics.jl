module Empirics

using Dates
using Lazy: @>
using DataFrames

import ..Geometry: mi

RADIUS = 50mi
ANALYSIS_DATES = Date(2016, 7, 1):Date(2019, 12, 31)
BLACKOUT_LENGTH = Quarter(2)

include("windows.jl")
include("registrations.jl")
include("demographics.jl")

end
