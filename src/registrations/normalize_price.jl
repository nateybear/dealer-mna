using DataStructures: DefaultDict
using OnlineStats: Mean, fit!, merge!
using Base: Fix1, Fix2
import ..snakemake
import JLD2

firstormissing(x, y) = all(ismissing, x) ? y : first(skipmissing(x))

normalize_trim(x...) = @> x begin
    firstormissing("")
    strip
    replace(r"[/-]" => " ")
    replace(r"[^\w\s]" => "")
    uppercase
    split(r"\s+")
    sort
    join(" ")
end

trimnames(df) = ["trim", "series", "trim2", "series2"] ∩ names(df)

function make_price_table()
    out = DefaultDict{Tuple{String, String, Int64, String}, Mean}(Mean)
    fitmean(v) = fit!(Mean(), v)
    for s in Arrow.Stream(snakemake.input["new_registrations"])
        s = DataFrame(s)
        transform!(s, trimnames(s) => ByRow(normalize_trim) => :extra)

        grouping = @> s begin
            groupby([:make, :model, :model_year, :extra])
            combine(:vehsalesprice => (fitmean ∘ skipmissing) => :stat)
        end
        for r in eachrow(grouping)
            merge!(out[(r.make, r.model, r.model_year, r.extra)], r.stat)
        end
    end

    out = DataFrame(
        make = getindex.(keys(out), 1),
        model = getindex.(keys(out), 2),
        model_year = getindex.(keys(out), 3),
        extra = getindex.(keys(out), 4),
        mean_price = getfield.(values(out), :μ),
        n = getfield.(values(out), :n),
    )

    sort!(out, [:make, :model, :model_year, :mean_price])
    subset!(out, :n => ByRow(>(5)))
    subset!(out, :extra => ByRow(!contains(r"\bWORK\b")))
    subset!(out, :extra => ByRow(!contains(r"\bFLEET\b")))
    subset!(out, :extra => ByRow(!contains(r"\bCOMMERCIAL\b")))
    subset!(out, :extra => ByRow(!contains(r"\bPOLICE\b")))

    evenly_spaced(x) = let n = length(x)
            range(0, 1, length = n + 1) |> (@$ Iterators.take(_, n)) |> collect
        end

    out = @> out begin
        groupby([:make, :model, :model_year])
        transform(
            :mean_price => minimum => :base_price,
            :mean_price => (x -> x ./ minimum(x)) => :trim_relprice,
            :mean_price => evenly_spaced => :trim_rank,
        )
    end

    return out
end

@memoize function price_table()
    if !isnothing(snakemake)
        return read_arrow(snakemake.input["price_table"])
    end

    return make_price_table()
end

function normalize_price(df, col_in = :vehsalesprice, col_out = :price_normed)
    return if isempty(df)
        hcat(
            df,
            DataFrame(
                :extra => String[],
                col_out => Float64[],
                ([:mean_price, :trim_rank, :trim_relprice] .=> Ref(Float64[]))...,
            ),
        )
    else
        @> df begin
            transform(trimnames(df) => ByRow(normalize_trim) => :extra)
            innerjoin(
                select(price_table(), Not(:n)), on = [:make, :model, :model_year, :extra]
            )
            transform(
                [col_in, :mean_price] => ByRow((p, bar) -> (p - bar) / bar) => col_out
            )
        end
    end
end
