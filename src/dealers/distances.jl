import ..Geometry: gdf, to_coords, distance
import ..Registrations: add_zips

@memoize shortest_distance(geometry, make) = minimum(
    x -> distance(geometry, x.geometry),
    eachrow(subset(State.df, :franchises => ByRow(∋(make))))
)

getgeom(dealer::Union{Dealership,DataFrameRow}) = dealer.geometry
getgeom(dealer::Tuple{Float64,Float64}) = dealer

using Unitful
function normed_distance(dealers, geometry, make)
    actual, i_shortest = findmin(distance(getgeom(dealer), geometry) for dealer in dealers)
    shortest = shortest_distance(geometry, make)
    miles(x) = ustrip(uconvert(u"mi", x))
    return (;
        actual_distance=miles(actual),
        shortest_distance=miles(shortest),
        normed_distance=actual / shortest,
        i_shortest=i_shortest,
    )
end

function add_distance(r, dealers)
    @> r add_zips transform([:geometry, :make] => ByRow(normed_distance$dealers) => AsTable)
end

function add_distance(r)
    r = add_zips(r)
    flat_dealers = @> State.df begin
        select(:geometry => :dealer_geometry, :gdns => :dlrgdn, :uid => :dealer_uid)
        flatten(:dlrgdn)
        groupby(:dlrgdn)
        combine([:dealer_geometry, :dealer_uid] .=> Ref{Vector}, renamecols=false)
    end
    r = innerjoin(r, flat_dealers, on=:dlrgdn, matchmissing=:notequal)
    transform(groupby(r, :dlrgdn)) do gdf
        dealers = first(gdf.dealer_geometry)
        @> gdf begin
            transform([:geometry, :make] => ByRow(normed_distance$dealers) => AsTable)
            transform([:dealer_uid, :i_shortest] => ByRow((x, y) -> x[y]) => :dealer_uid)
            select(Not(:dealer_geometry, :i_shortest))
        end
    end
end

add_distance(r, t::Transaction) = add_distance(r, eachrow(State.df[dealer_idxs(t), :]))
