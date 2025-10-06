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

#### run daemon inside of this REPL in the background (for shell commands)
import DaemonMode
import Base.Threads: @spawn
daemon = nothing
start_daemon() =
    let daemon = getproperty(@__MODULE__, :daemon)
        if isnothing(daemon) || istaskdone(daemon)
            @info "Starting daemon"
            daemon = @spawn DaemonMode.serve(async = true)
            setproperty!(@__MODULE__, :daemon, daemon)
            sleep(0.5)  # give the daemon some time to start
            istaskfailed(daemon) && error("Daemon failed to start")
            return daemon
        else
            @info "Daemon already running"
            return daemon
        end
    end

function stop_daemon()
    daemon = getproperty(@__MODULE__, :daemon)
    if !isnothing(daemon) && !istaskdone(daemon)
        @info "Stopping daemon"
        istaskdone(daemon) || DaemonMode.sendExitCode()
        sleep(0.5)  # give the daemon some time to stop
        istaskdone(daemon) || error("Daemon did not stop properly")
    else
        @info "No daemon running"
    end
    setproperty!(@__MODULE__, :daemon, nothing)
    return nothing
end

function here(path...)
    base_folder = dirname(Base.current_project())
    return joinpath(base_folder, path...)
end

module Limiter

export RateLimiter

using Dates: DateTime, now, Millisecond
using HTTP
using Unitful

const 𝐓⁻¹ = Unitful.Dimensions{(Unitful.Dimension{:Time}(-1//1),)}()
const Rate = Unitful.Quantity{Int64, 𝐓⁻¹, <:Any}

mutable struct TokenBucket
    last::DateTime
    count::Int64
    rate::Rate
    TokenBucket(rate::Rate) = new(now(), 0, rate)
end

function ask(bucket::TokenBucket)
    update_window = bucket.rate.val / bucket.rate
    if now() - bucket.last > update_window
        prev_last = bucket.last
        bucket.count = 1
        bucket.last = now()
        true, prev_last
    elseif bucket.count < bucket.rate.val
        bucket.count += 1
        true, bucket.last
    else
        false, bucket.last + update_window
    end
end

struct RateLimiter{N}
    buckets::NTuple{N, TokenBucket}
    function RateLimiter(buckets::Vararg{Rate})
        return new{length(buckets)}(Tuple(TokenBucket(b) for b in buckets))
    end
end

function ask(limiter::RateLimiter)
    reponded_ok, prev_last = zip(ask.(limiter.buckets)...)
    if !all(reponded_ok)
        for idx_to_reset in findall(reponded_ok)
            b = limiter.buckets[idx_to_reset]
            b.count -= 1
            b.last = prev_last[idx_to_reset]
        end
        false, maximum(prev_last[findall(.!reponded_ok)])
    else
        true, nothing
    end
end

backoff(limiter) =
    while true
        can_i, must_wait = ask(limiter)
        can_i && break
        jitter = Millisecond(round(Int64, 1000rand()))
        sleep(abs(must_wait - now()) + jitter)  # add some jitter to avoid thundering herd problem
    end

function HTTP.get(limiter::RateLimiter, args...; kwargs...)
    backoff(limiter)
    return HTTP.get(args...; kwargs...)
end

function HTTP.post(limiter::RateLimiter, args...; kwargs...)
    backoff(limiter)
    return HTTP.post(args...; kwargs...)
end

end

##### support mapreduce with grouped dataframes
import SplittablesBase

function SplittablesBase.halve(gd::GroupedDataFrame)
    let (one, two) = SplittablesBase.halve(keys(gd))
        (gd[one], gd[two])
    end
end


pluck(df, args...; kwargs...) = only(eachrow(subset(df, args...; kwargs..., view = true)))
