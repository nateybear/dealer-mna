module DealerMnA

using Reexport

@reexport begin
    using DataFrames
    using Arrow
    using CSV
    using UUIDs
    using PartialFunctions
    using Lazy: @>
    import JLD2
    using Missings
    using Dates
    using Cleaner
end

export Geometry,
    Registrations, Dealers, Empirics, Demand, snakemake, snakemake!, with_logging, nonmissing,
    read_csv, read_arrow, start_daemon, stop_daemon, pluck

using Random
_rng = MersenneTwister(78735)
const TRANSITION_GID = uuid4(_rng)
const SINGLE_OPERATOR_GID = uuid4(_rng)

include("utils.jl")
include("geometry/Geometry.jl")
include("registrations/Registrations.jl")
include("dealers/Dealers.jl")
include("empirics/Empirics.jl")
include("demand/Demand.jl")

# attempt to force LSP to recognize module symbols in scripts
@static if false
    for file in readdir("../scripts")
        include("../scripts/$file")
    end
end

end
