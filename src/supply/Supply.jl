module Supply

using Memoize
using UUIDs

using ..Demand
import ..Demand: QuarterKey, VehicleKey

include("state.jl")
include("xgboost.jl")
include("panel.jl")
include("nonparametric.jl")

end
