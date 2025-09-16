using DealerMnA
using Plots
using StatsBase
using Loess
using Splines2

!isempty(ARGS) && snakemake!(ARGS[1])

function all_regressions(group_df)
    all_arrow_files = map(eachrow(group_df)) do r
        [
            "data/treated/$(r.group)/last_registrations_before_$(r.transaction_uid)_$(r.dealer_uid).arrow",
            "data/treated/$(r.group)/first_registrations_after_$(r.transaction_uid)_$(r.dealer_uid).arrow",
            "data/control/$(r.group)/last_registrations_before_$(r.transaction_uid)_$(r.dealer_uid).arrow",
            "data/control/$(r.group)/first_registrations_after_$(r.transaction_uid)_$(r.dealer_uid).arrow",
        ]
    end
    return DataFrame(Arrow.Table(collect(Iterators.flatten(all_arrow_files))))
end

function splines(x)
    return ns(
        collect(x),
        knots = [-0.1, -0.05, -0.01, 0, 0.01, 0.05, 0.1],
        intercept = true,
    )
end
function addsplines!(df, col = :change_market_share)
    S = splines(df[!, col])
    for i in 1:size(S, 2)
        df[:, "ns_$(i)"] = S[:, i]
    end
    return nothing
end

parseindex(s) = parse(Int64, replace(s, "ns_" => ""))

function calc_se(V, idx, values)
    ∇ = zeros(size(V, 1))
    for (i, val) in zip(idx, values)
        ∇[i] = val
    end
    return sqrt(only(∇'V * ∇))
end

with_logging() do
    df = DataFrame(Arrow.Table(snakemake.input["group_market_share_changes"]))
    df.reg_uid = df.transaction_uid .* "_" .* df.dealer_uid
    col = Symbol(snakemake.wildcards["col"])
    coeftable = DataFrame(Arrow.Table(snakemake.input["treatment_effect"]))
    diffs = @> coeftable begin
        subset(:T => ByRow(in([-3, 2])))
        sort(:T, rev = true)
        groupby(:interaction)
        combine(col => splat(-) => :diff)
    end
    extras = JLD2.load_object(snakemake.input["extra_stats"])
    col_extras = only(eachrow(subset(extras, :stat => ByRow(==(col)))))

    # covariance matrix of all the differences
    V = let c = reshape(unique(coeftable.interaction), 1, :)
        spline_mask = coeftable.interaction .== c
        ∇ = (-(coeftable.T .== -3) .+ (coeftable.T .== 2)) .* spline_mask
        ∇'col_extras.V * ∇
    end

    x_grid = DataFrame(x = range(extrema(df.change_market_share)..., length = 500))
    addsplines!(x_grid, :x)

    fitted_values = @> x_grid begin
        stack(names(x_grid, r"ns_"))
        innerjoin(diffs, on = :variable => :interaction)
        transform(
            [:value, :diff] => ByRow(*) => :contribution, 
            :variable => ByRow(parseindex) => :idx
        )
        groupby(:x)
        combine(:contribution => sum => :y, [:idx, :value] => calc_se$V => :se)
    end

    df = @> df begin
        subset(:change_market_share => ByRow(!iszero)) 
        innerjoin(col_extras.fes, on = :reg_uid)
    end
    scatter(
        df.change_market_share,
        df.diff_in_diff,
        label = nothing,
        alpha = 0.5,
        size = (800, 600),
        xlabel = "Change in Market Share",
        ylabel = "Change in $(replace(string(col), '_' => ' ') |> titlecase)",
        left_margin = 5Plots.mm,
    )
    hline!([0.0], color = "black", label = nothing)
    vline!([0.0], color = "black", label = nothing)
    
    plot!(
        fitted_values.x, 
        fitted_values.y, 
        ribbon=1.96fitted_values.se, 
        msc = 2, 
        label = nothing, 
        linestrokewidth = 4,
        alpha = 0.8,
    )

    png(snakemake.output[1])
end
