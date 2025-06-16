module Empirics

using Dates
using Lazy: @>
using DataFrames

import ..Geometry: mi

RADIUS = 50mi
ANALYSIS_DATES = Date(2016, 1, 1):Date(2019, 12, 31)
BLACKOUT_LENGTH = Quarter(4)
PERIOD_LENGTH = Quarter(1)
VALID_PERIODS = -4:3

include("windows.jl")
include("registrations.jl")
include("demographics.jl")
include("mahalanobis.jl")
include("event_study.jl")
include("treatment_effect.jl")
include("combine_std_error.jl")
include("synthetic_comparison.jl")
include("market_share.jl")

end
