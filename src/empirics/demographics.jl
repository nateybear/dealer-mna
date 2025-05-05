### this is demographics for DEALERSHIPS, using weighted averages of zip demographics

using StatsBase: weights as w, mean
using ..Geometry: distance
using ..Registrations: zips
using ..Dealers: State
using UUIDs

struct UniformWeight end
(::UniformWeight)(_) = 1
@kwdef struct LinearWeight{T}
    max_distance::T = 50mi
end
(l::LinearWeight)(d) = max(0, 1 - d / l.max_distance)

total_population(id::UUID, distance_weight=LinearWeight()) =
    let idx = findfirst(==(id), State.df.uid)
        return total_population(State.df[idx, :], distance_weight)
    end
function total_population(d, distance_weight=LinearWeight())
    z = transform(zips(), :geometry => ByRow(g -> distance(g, d.geometry)) => :distance)
    subset!(z, :distance => ByRow(x -> x < RADIUS))
    return sum(z.population .* distance_weight.(z.distance))
end

pop_weighted(id::UUID, col, distance_weight=LinearWeight()) =
    let idx = findfirst(==(id), State.df.uid)
        return pop_weighted(State.df[idx, :], col, distance_weight)
    end
function pop_weighted(d, col, distance_weight=LinearWeight())
    z = transform(zips(), :geometry => ByRow(g -> distance(g, d.geometry)) => :distance)
    subset!(z, :distance => ByRow(x -> x < RADIUS))
    dropmissing!(z, col)
    return mean(z[!, col], w(z.population .* distance_weight.(z.distance)))
end

median_income(x, distance_weight=LinearWeight()) = pop_weighted(x, :median_earnings, distance_weight)
share_urban(x, distance_weight=LinearWeight()) = pop_weighted(x, :urban_pct, distance_weight)
share_white(x, distance_weight=LinearWeight()) = pop_weighted(x, :white_pct, distance_weight)
share_black(x, distance_weight=LinearWeight()) = pop_weighted(x, :black_pct, distance_weight)
share_hispanic(x, distance_weight=LinearWeight()) = pop_weighted(x, :hispanic_pct, distance_weight)
