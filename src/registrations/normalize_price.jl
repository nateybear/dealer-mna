using DataStructures: DefaultDict
using OnlineStats: Mean, fit!, merge!
using Base: Fix1, Fix2

normalize_trim = ByRow() do x...
    trim_vars = reduce(skipmissing(x), init=[""]) do acc, trim
        split_trim = @> trim strip replace(r"[^\w\s]" => "") uppercase split(r"\s+")
        union(acc, split_trim)
    end

    @> trim_vars sort join(" ") strip
end

trimnames(df) = names(df, r"^((trim)|(series))$")

@memoize function price_table()
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

    return out
end

normalize_price(df, col_in=:vehsalesprice, col_out=:price_normed) =
    isempty(df) ?
    hcat(df, DataFrame(:extra => String[], col_out => Float64[], :mean_price => Float64[])) :
    @> df begin
        transform(trimnames(df) => normalize_trim => :extra)
        transform([col_in, :make, :model, :extra] =>
            ByRow() do price, make, model, extra
                mean_price = price_table()[(make, model, extra)].μ
                (; mean_price, col_out => (price - mean_price) / mean_price)
            end => AsTable
        )
    end
