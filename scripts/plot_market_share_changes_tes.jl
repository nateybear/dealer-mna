using DealerMnA
using Plots
using StatsBase
using Splines2

!isempty(ARGS) && snakemake!(ARGS[1])

const NS = Ref{Function}()

function calcsplines!(x)
    if !isdefined(NS, 1)
        NS[] = ns_(
            collect(x),
            boundary_knots = (-0.1, 0.1),
            df = 5,
            intercept = true,
        )
    end
    return NS[]
end

function addsplines!(df, col = :change_market_share)
    f = calcsplines!(df[!, col])
    S = f(df[!, col])
    for (i, col) in enumerate(eachcol(S))
        df[:, "ns_$(i)"] = col
    end
    return nothing
end

parseindex(s) = parse(Int64, match(r"^ns_(\d).*$", s).captures[1])
parsespline(s) = "ns_$(parseindex(s))"

function calc_se(V, idx, values)
    ∇ = zeros(size(V, 1))
    for (i, val) in zip(idx, values)
        if !ismissing(val)
            ∇[i] = val
        end
    end
    return sqrt(only(∇'V * ∇))
end

difference(x) = vcat(diff(x), missing)

test_deriv(estimate, std_err) =
    if estimate < 0
        estimate < -1.96std_err ? :neg : :zero
    elseif estimate > 1.96std_err
        :pos
    else
        :zero
    end

"""
    spans_from_labels(xs::Vector, labels::Vector)

Given a grid `xs` and categorical `labels` (e.g. :pos, :neg, :zero),
return a vector of `(a, b, label)` tuples representing consecutive
intervals where the label is constant.
"""
function spans_from_labels(xs::Vector, labels::Vector)
    @assert length(xs) == length(labels)

    spans = []
    start = xs[1]
    current_label = labels[1]

    for i in 2:length(xs)
        if labels[i] != current_label
            # close span at previous point
            push!(spans, (start, xs[i - 1], current_label))
            # start new span
            start = xs[i]
            current_label = labels[i]
        end
    end

    # close last span
    push!(spans, (start, xs[end], current_label))

    return spans
end

with_logging() do
    df = read_arrow(snakemake.input["group_market_share_changes"])
    calcsplines!(df.change_market_share)
    col = Symbol(snakemake.wildcards["col"])
    tes = JLD2.load_object(snakemake.input["treatment_effect"])[col]

    x_grid = DataFrame(x = range(extrema(df.change_market_share)..., length = 500))
    addsplines!(x_grid, :x)
    transform!(x_grid, names(x_grid, r"ns_") .=> difference)

    fitted_values = @> x_grid begin
        stack(names(x_grid, r"ns_"))
        transform(
            :variable => ByRow(parsespline) => :spline,
            :variable => ByRow(contains("difference")) => :isdifference,
        )
        innerjoin(diffs, on = :spline => :interaction)
        transform(
            [:value, :diff] => ByRow(passmissing(*)) => :contribution,
            :variable => ByRow(parseindex) => :idx,
        )
        groupby([:x, :isdifference])
        combine(
            :contribution => sum ∘ skipmissing => :y, [:idx, :value] => calc_se$V => :se
        )
    end

    df = @> df begin
        subset(:change_market_share => ByRow(!iszero))
        innerjoin(col_extras.fes, on = :reg_uid)
    end
    scatter(
        df.change_market_share,
        df.diff_in_diff,
        label = nothing,
        size = (800, 600),
        xlabel = "Change in Market Share",
        ylabel = "Change in $(replace(string(col), '_' => ' ') |> titlecase)",
        left_margin = 5Plots.mm,
    )

    deriv = subset(fitted_values, :isdifference => ByRow(identity))
    deriv = deriv[1:(end - 1), :] # drop last value (nothing)
    transform!(deriv, [:y, :se] => ByRow(test_deriv) => :sign)
    shaded_regions = spans_from_labels(deriv.x, deriv.sign)

    for (xstart, xstop, sign) in shaded_regions
        @show xstart xstop sign
        if sign == :pos
            @info "POSSS"
            vspan!([xstart, xstop], alpha = 0.1, color = :green, label = nothing)
        elseif sign == :neg
            @info "NEGGG"
            vspan!([xstart, xstop], alpha = 0.1, color = :red, label = nothing)
        end
    end

    hline!([0.0], color = "black", label = nothing)
    vline!([0.0], color = "black", label = nothing)

    fitted = subset(fitted_values, :isdifference => ByRow(!))
    plot!(
        fitted.x,
        fitted.y,
        ribbon = 1.96fitted_values.se,
        color = :grey,
        label = nothing,
        linestrokewidth = 4,
        alpha = 0.4,
    )

    png(snakemake.output[1])
end
