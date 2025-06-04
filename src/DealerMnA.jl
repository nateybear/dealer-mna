module DealerMnA

using Reexport

@reexport begin
    using DataFrames
    using Arrow
    using CSV
    using PartialFunctions
    using Lazy: @>
    import JLD2
    using Missings
    using Dates
    using Cleaner
end

export Geometry, Registrations, Dealers, Empirics, snakemake, with_logging, nonmissing

include("utils.jl")
include("geometry/Geometry.jl")
include("registrations/Registrations.jl")
include("dealers/Dealers.jl")
include("empirics/Empirics.jl")

### speed up startup time
module Startup
    using PrecompileTools: @setup_workload, @compile_workload
    using DrWatson
    import JLD2
    import Arrow
    using DataFrames
    import Memoize
    import ..Dealers, ..Empirics, ..Registrations, ..nonmissing

    @setup_workload begin
        if isfile(datadir("transactions_for_analysis.jld2")) &&
            isfile(datadir("dealers_init.jld2"))
            @compile_workload begin
                Dealers.load_from_disk!(datadir("dealers_init.jld2"))
                t = JLD2.load_object(datadir("transactions_for_analysis.jld2"))[1]
                control_dealers = Dealers.timetravel(t) do
                    reduce(union, Empirics.control_dealers(t).gdns)
                end
                try
                    df = subset(
                        Registrations.registrations(),
                        :dlrgdn => ByRow(nonmissing(in(control_dealers))),
                    )
                    df = transform(df,
                        :vehsalesprice => ByRow(x -> x / 1000) => :vehsalesprice,
                    )
                    Arrow.write("temp.arrow", df)
                    rm("temp.arrow")
                    empty!(Memoize.memoize_cache(Registrations.registrations)) # force each julia process to mmap their own registrations
                catch
                    @warn "Loading registrations failed, skipping precompilation"
                end
            end
        end
    end
end

end
