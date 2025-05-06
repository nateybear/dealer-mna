module DealerMnA

export Geometry, Registrations, Dealers, Empirics, snakemake, with_logging

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
import ..Dealers, ..Empirics

@setup_workload begin
    if isfile(datadir("transactions_for_analysis.jld2")) && isfile(datadir("dealers_init.jld2"))
        @compile_workload begin
            Dealers.load_from_disk!(datadir("dealers_init.jld2"))
            t = JLD2.load_object(datadir("transactions_for_analysis.jld2"))[1]
            df = Empirics.last_registrations_before(t)
            df = subset(df, :make => ByRow(==("FORD")))
            df = transform(df, :vehsalesprice => ByRow(x -> x / 1000) => :vehsalesprice)
            Arrow.write("temp.arrow", df)
            rm("temp.arrow")
        end
    end
end
end

end
