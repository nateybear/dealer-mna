module Demand

import ..Registrations
import ..Dealers
import ..Geometry: distance, miles

using UUIDs
using PartialFunctions
using DataFrames
using Lazy: @>
using Random
using LogExpFunctions
using Distributions
using NamedArrays
import Base.Iterators: partition
import Dates
using DataStructures: DefaultDict

_rng = MersenneTwister(78735)
const TRANSITION_GID = uuid4(_rng)
const SINGLE_OPERATOR_GID = uuid4(_rng)

include("loading.jl")
include("models.jl")

end
