using DataStructures: DefaultDict
using OnlineStats: Mean, fit!, merge!
using Base: Fix1, Fix2
import ..snakemake
import JLD2

normalize_trim = ByRow() do x...
    trim_vars = reduce(skipmissing(x), init=[""]) do acc, trim
        split_trim = @> trim strip replace(r"[^\w\s]" => "") uppercase split(r"\s+")
        union(acc, split_trim)
    end

    @> trim_vars sort join(" ") strip
end

trimnames(df) = names(df, r"^((trim)|(series))$")

@memoize function price_table()
    if !isnothing(snakemake) && haskey(snakemake.input, "price_table")
        return DataFrame(Arrow.Table(snakemake.input["price_table"]))
    end

    out = DefaultDict{NTuple{3,String},Mean}(Mean)
    fitmean(v) = fit!(Mean(), v)
    for s in Arrow.Stream(datadir("arrow/new_registrations.arrow"))
        s = DataFrame(s)
        transform!(
            s,
            trimnames(s) => normalize_trim => :extra
        )

        grouping = @> s groupby([:make, :model, :extra]) combine(:vehsalesprice => (fitmean ∘ skipmissing) => :stat)
        for r in eachrow(grouping)
            merge!(out[(r.make, r.model, r.extra)], r.stat)
        end
    end

    out = DataFrame(
        make=getindex.(keys(out), 1),
        model=getindex.(keys(out), 2),
        extra=getindex.(keys(out), 3),
        mean_price=getfield.(values(out), :μ),
        n=getfield.(values(out), :n),
    )

    sort!(out, [:make, :model, :mean_price])
    subset!(out, :n => ByRow(>(10)))

    out = @> out begin
        groupby([:make, :model])
        transform(
            :mean_price => (x -> x ./ minimum(x)) => :trim_relprice,
            :mean_price => (x -> length(x) == 1 ? [0.0] : collect(range(0, 1, length=length(x)))) => :trim_rank
        )
    end

    return out
end

normalize_price(df, col_in=:vehsalesprice, col_out=:price_normed) =
    isempty(df) ?
    hcat(df, DataFrame(:extra => String[], col_out => Float64[], [:mean_price, :trim_rank, :trim_relprice] .=> Float64[])) :
    @> df begin
        transform(trimnames(df) => normalize_trim => :extra)
        innerjoin(select(price_table(), Not(:n)), on=[:make, :model, :extra])
        transform([col_in, :mean_price] => ByRow((p, bar) -> (p - bar) / bar) => col_out)
    end
