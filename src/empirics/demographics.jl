### this is demographics for DEALERSHIPS, using weighted averages of zip demographics

using StatsBase: weights as w, mean
using ..Geometry: distance
using ..Registrations: zips
using ..Dealers: State
using UUIDs

struct UniformWeight end
(::UniformWeight)(_) = 1
@kwdef struct LinearWeight{T}
    max_distance::T = 200mi
end
(l::LinearWeight)(d) = max(0, 1 - d / l.max_distance)

max_distance(::UniformWeight) = 10_000mi # Oh I would walk 10,000 miles...
max_distance(l::LinearWeight) = l.max_distance

function total_population(id::UUID, distance_weight = LinearWeight())
    idx = findfirst(==(id), State.df.uid)
    return total_population(State.df[idx, :], distance_weight)
end
function total_population(d, distance_weight = LinearWeight())
    z = transform(zips(), :geometry => ByRow(g -> distance(g, d.geometry)) => :distance)
    z = subset(z, :distance => ByRow(x -> x < RADIUS))
    return sum(z.population .* distance_weight.(z.distance))
end

function pop_weighted(id::UUID, col, distance_weight = LinearWeight())
    idx = findfirst(==(id), State.df.uid)
    return pop_weighted(State.df[idx, :], col, distance_weight)
end
function pop_weighted(d, col, distance_weight = LinearWeight())
    z = transform(zips(), :geometry => ByRow(g -> distance(g, d.geometry)) => :distance)
    z = subset(z, :distance => ByRow(x -> x < RADIUS))
    z = dropmissing(z, col)
    return mean(z[!, col], w(z.population .* distance_weight.(z.distance)))
end

function change_population(x, distance_weight = LinearWeight())
    return pop_weighted(x, :population_pct_change, distance_weight)
end

function median_income(x, distance_weight = LinearWeight())
    return pop_weighted(x, :median_earnings, distance_weight)
end
function share_urban(x, distance_weight = LinearWeight())
    return pop_weighted(x, :urban_pct, distance_weight)
end
function share_white(x, distance_weight = LinearWeight())
    return pop_weighted(x, :white_pct, distance_weight)
end
function share_black(x, distance_weight = LinearWeight())
    return pop_weighted(x, :black_pct, distance_weight)
end
function share_hispanic(x, distance_weight = LinearWeight())
    return pop_weighted(x, :hispanic_pct, distance_weight)
end
