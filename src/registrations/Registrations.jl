module Registrations

using DrWatson
using DataFrames
using Arrow
using CSV
using Memoize
using PartialFunctions
using Lazy: @>

import ..snakemake

@memoize registrations() =
    let arrow_file = datadir("arrow/new_registrations.arrow"),
        fname = if (isnothing(snakemake) || Base.isinteractive()) && isfile(arrow_file) # this is during package init or REPL usage
            arrow_file
        elseif haskey(snakemake.input, "new_registrations") # otherwise expect snakemake to be properly instantiated
            snakemake.input["new_registrations"]
        else
            error("Arrow file for new registrations not found: $arrow_file")
        end

        DataFrame(Arrow.Table(fname))
    end

include("zips.jl")
include("normalize_price.jl")
include("normalize_body_class.jl")
include("loading.jl")

end
