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
    Registrations, Dealers, Empirics, snakemake, snakemake!, with_logging, nonmissing,
    read_csv, start_daemon, stop_daemon, pluck

include("utils.jl")
include("geometry/Geometry.jl")
include("registrations/Registrations.jl")
include("dealers/Dealers.jl")
include("empirics/Empirics.jl")

@static if false
    for file in readdir("../scripts")
        include("../scripts/$file")
    end
end

end
