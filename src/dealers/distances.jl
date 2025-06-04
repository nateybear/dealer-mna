import ..Geometry: gdf, to_coords, distance
import ..Registrations: add_zips

@memoize shortest_distance(geometry, make) = minimum(
    x -> distance(geometry, x.geometry),
    eachrow(subset(State.df, :franchises => ByRow(∋(make)))),
)

getgeom(dealer::Union{Dealership, DataFrameRow}) = dealer.geometry
getgeom(dealer::Tuple{Float64, Float64}) = dealer

using Unitful
function normed_distance(dealers, geometry, make)
    actual, i_shortest = findmin(distance(getgeom(dealer), geometry) for dealer in dealers)
    shortest = shortest_distance(geometry, make)
    miles(x) = ustrip(uconvert(u"mi", x))
    return (;
        actual_distance = miles(actual),
        shortest_distance = miles(shortest),
        normed_distance = actual / shortest,
        i_shortest = i_shortest,
    )
end

function add_distance(r, dealers)
    @> r add_zips transform([:geometry, :make] => ByRow(normed_distance$dealers) => AsTable)
end

function add_distance(r)
    r = add_zips(r)
    isempty(r) && return missing
    r.uid = [uuid4() for _ in 1:nrow(r)]
    flat_dealers = @> State.df begin
        select(
            :geometry => :dealer_geometry,
            :gdns => :dlrgdn,
            :uid => :dealer_uid,
            :franchises => :make,
        )
        flatten(:dlrgdn)
        flatten(:make)
    end
    r = innerjoin(r, flat_dealers, on = [:dlrgdn, :make], matchmissing = :notequal)
    isempty(r) && return missing
    combine(groupby(r, :uid), keepkeys = false) do gdf
        dealers = gdf.dealer_geometry
        geometry = first(gdf.geometry)
        make = first(gdf.make)
        dist_info = normed_distance(dealers, geometry, make)
        @> gdf[[dist_info.i_shortest], :] begin
            hcat(DataFrame(; dist_info...))
            select(Not(:dealer_geometry, :i_shortest))
        end
    end
end

add_distance(r, t::Transaction) = add_distance(r, eachrow(State.df[dealer_idxs(t), :]))
