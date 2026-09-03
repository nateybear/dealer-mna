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
    Registrations, Dealers, Empirics, Demand, Supply, snakemake, snakemake!, with_logging,
    nonmissing,
    read_csv, read_arrow, start_daemon, stop_daemon, pluck

using Random
_rng = MersenneTwister(78735)
const TRANSITION_GID = uuid4(_rng)
const SINGLE_OPERATOR_GID = uuid4(_rng)

# Pin Julia's worker threads to physical cores at package load. Without this the
# OS is free to migrate threads across cores, which shows up as jitter in the
# @turbo / @batch hot loops (softmax, forward_pass, residual_demand) — stable
# affinity makes per-thread cache and BLAS behavior reproducible. Wrapped in a
# try/catch so package load doesn't fail on platforms where affinity isn't
# available (e.g. macOS just emits a warning).
using ThreadPinning
function __init__()
    try
        pinthreads(:cores)
    catch e
        @warn "ThreadPinning.pinthreads(:cores) failed — continuing unpinned" exception = e
    end
end

include("utils.jl")
include("geometry/Geometry.jl")
include("registrations/Registrations.jl")
include("dealers/Dealers.jl")
include("empirics/Empirics.jl")
include("demand/Demand.jl")
include("supply/Supply.jl")

# attempt to force LSP to recognize module symbols in scripts
@static if false
    for file in readdir("../scripts")
        include("../scripts/$file")
    end
end

end
