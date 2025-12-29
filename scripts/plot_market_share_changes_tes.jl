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

function calc_se(tes, idx, values)
    labels = replace.(idx, "_difference" => "")
    μ = tes.did_estimate[labels].array
    V = tes.did_stderr[labels, labels].array
    return (
        est = μ'values,
        se = sqrt(values'V * values),
    )
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
    df = @> read_arrow(snakemake.input["group_market_share_changes"]) begin
        transform([:dealer_uid, :transaction_uid] .=> ByRow(UUID), renamecols = false)
        groupby(:dealer_uid)
        subset(:change_market_share => (x -> abs.(x) .== maximum(abs, x)))
        unique(:dealer_uid) # if multiple maxima select the first
    end
    calcsplines!(df.change_market_share)
    col = Symbol(snakemake.wildcards["col"])
    tes = JLD2.load_object(snakemake.input["treatment_effect"])[col]

    x_grid = DataFrame(x = range(1.05 .* extrema(df.change_market_share)..., length = 500))
    addsplines!(x_grid, :x)
    transform!(x_grid, names(x_grid, r"ns_") .=> difference)
    dropmissing!(x_grid)

    fitted_values = @> x_grid begin
        stack(names(x_grid, r"ns_"))
        transform(
            :variable => ByRow(contains("difference")) => :isdifference
        )
        groupby([:x, :isdifference])
        combine(
            [:variable, :value] => (calc_se$tes) => AsTable
        )
    end

    df = innerjoin(df, tes.extras, on = :dealer_uid)

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
    transform!(deriv, [:est, :se] => ByRow(test_deriv) => :sign)
    shaded_regions = spans_from_labels(deriv.x, deriv.sign)
    
    avg_deriv = mapreduce(vcat, eachrow(df)) do r
        i = searchsortedfirst(x_grid.x, r.change_market_share)
        return x_grid[[i], :]
    end
    avg_deriv = combine(
        avg_deriv, 
        names(avg_deriv, r"_difference") .=> mean, 
        renamecols = false
    )
    avg_deriv = @> avg_deriv stack() combine([:variable, :value] => (calc_se$tes) => AsTable) round.(digits=4) getindex(1, :)
        
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
        fitted.est,
        ribbon = 1.96fitted_values.se,
        color = :grey,
        label = nothing,
        linestrokewidth = 4,
        alpha = 0.4,
    )

    # find the most empty corner to place the average derivative annotation
    x_pos = extrema(df.change_market_share)
    y_pos = extrema(df.diff_in_diff)
    corners = collect(Iterators.product(x_pos, y_pos))
    text_positions = collect(Iterators.product(
        (:left, :right),
        (:bottom, :top),
    ))
    counts = map(corners) do (x_c, y_c)
        x_counts = x_c < mean(df.change_market_share) ?
            df.change_market_share .<= mean(df.change_market_share) :
            df.change_market_share .>= mean(df.change_market_share)

        y_counts = y_c < mean(df.diff_in_diff) ?
            df.diff_in_diff .<= mean(df.diff_in_diff) :
            df.diff_in_diff .>= mean(df.diff_in_diff)

        return sum(x_counts .& y_counts)
    end
    best_corner = corners[argmin(counts)]
    best_text_pos = text_positions[argmin(counts)]

    annotate!(
        best_corner...,
        text(
            "Avg. Deriv.: $(avg_deriv.est) ($(avg_deriv.se))",
            12,
            :black,
            best_text_pos...,
        ),
    )

    png(snakemake.output[1])
end
