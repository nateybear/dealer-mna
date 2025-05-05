#### parse snakemake input
import JSON3

@kwdef struct Snakemake
    input::Union{Dict,Vector} = Dict()
    output::Union{Dict,Vector} = Dict()
    params::Union{Dict,Vector} = Dict()
    wildcards::Union{Dict,Vector} = Dict()
    log::Union{Dict,Vector} = Dict()
    rule::String = ""
end

snakemake = nothing

function __init__()
    global snakemake
    in_dict = Base.isinteractive() ? Dict() : JSON3.read(stdin, Dict)
    snakemake = Snakemake(; (Symbol(k) => v for (k, v) in in_dict)...)
    return nothing
end

#### wrap logging
import LoggingExtras
import Logging

with_logging(f) = LoggingExtras.with_logger(
    f,
    LoggingExtras.MinLevelLogger(
        LoggingExtras.FileLogger(snakemake.log[1]),
        Logging.Info
    )
)

### nonmissing helper
struct NonMissing{F}
    f::F
end
nonmissing(f) = NonMissing(f)
(::NonMissing)(::Missing) = false
((; f)::NonMissing)(x) = f(x)
