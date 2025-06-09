#### parse snakemake input
import JSON3

@kwdef struct Snakemake
    input::Union{Dict, Vector} = Dict()
    output::Union{Dict, Vector} = Dict()
    params::Union{Dict, Vector} = Dict()
    wildcards::Union{Dict, Vector} = Dict()
    log::Union{Dict, Vector} = Dict()
    rule::String = ""
end

snakemake = nothing

function snakemake!(infile = stdin)
    in_dict = infile == stdin && Base.isinteractive() ? Dict() : JSON3.read(infile, Dict)
    snek = Snakemake(; (Symbol(k) => v for (k, v) in in_dict)...)
    setproperty!(@__MODULE__, :snakemake, snek)
    return nothing
end

__init__() = snakemake!()

#### wrap logging
import LoggingExtras
import Logging

function with_logging(f)
    return LoggingExtras.with_logger(
        f,
        LoggingExtras.MinLevelLogger(
            LoggingExtras.FileLogger(snakemake.log[1]),
            Logging.Info,
        ),
    )
end

### nonmissing helper
struct NonMissing{F}
    f::F
end
nonmissing(f) = NonMissing(f)
(::NonMissing)(::Missing) = false
((; f)::NonMissing)(x) = f(x)

### read_csv helper
read_csv = @$ CSV.read(_, DataFrame)
