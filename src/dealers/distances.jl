import ..Geometry: gdf, to_coords, miles
import ..Registrations: add_zips
import Folds

@memoize shortest_distance(geometry, make) = minimum(
    x -> miles(geometry, x.geometry),
    eachrow(subset(State.df, :franchises => ByRow(∋(make)), :active => ByRow(identity))),
)

getgeom(dealer::Union{Dealership, DataFrameRow}) = dealer.geometry
getgeom(dealer::Tuple{Float64, Float64}) = dealer

function dealer_distance(dealers, geometry, make; normalize)
    actual_distance, i_shortest = findmin(miles(getgeom(dealer), geometry) for dealer in dealers)
    shortest = normalize && shortest_distance(geometry, make)
    return (;
        actual_distance,
        i_shortest,
        (normalize ? [:shortest_distance => shortest, :normed_distance => actual_distance / shortest] : [])...,
    )
end

function add_distance(r, dealers; normalize=true)
    @> r add_zips transform([:geometry, :make] => ByRow(dealer_distance$(dealers; normalize)) => AsTable)
end

function add_distance(r; normalize=true)
    r = add_zips(r)
    isempty(r) && return missing
    N = nrow(r)
    r.uid = [uuid4() for _ in 1:N] # add a unique identifier for each row to group back together later
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
    return combine(groupby(r, :uid), keepkeys=false) do gdf
        dealers = gdf.dealer_geometry
        geometry = first(gdf.geometry)
        make = first(gdf.make)
        dist_info = dealer_distance(dealers, geometry, make; normalize)
        @> gdf[[dist_info.i_shortest], :] begin
            hcat(DataFrame(; dist_info...))
            select(Not(:dealer_geometry, :i_shortest))
        end
    end
end

add_distance(r, t::Transaction) = add_distance(r, eachrow(State.df[dealer_idxs(t), :]))
