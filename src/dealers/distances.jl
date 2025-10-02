import ..Geometry: gdf, to_coords, distance
import ..Registrations: add_zips
import Folds

@memoize shortest_distance(geometry, make) = minimum(
    x -> distance(geometry, x.geometry),
    eachrow(subset(State.df, :franchises => ByRow(∋(make)), :active => ByRow(identity))),
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

function add_distance(r; state = State.df)
    r = add_zips(r)
    isempty(r) && return missing
    N = nrow(r)
    r.uid = [uuid4() for _ in 1:N] # add a unique identifier for each row to group back together later
    flat_dealers = @> state begin
        subset(:active => ByRow(identity))
        select(
            :geometry => :dealer_geometry,
            :gdns => :dlrgdn,
            :uid => :dealer_uid,
            :franchises => :make,
            :gid => :dealer_gid,
        )
        flatten(:dlrgdn)
        flatten(:make)
    end
    r = innerjoin(r, flat_dealers, on = [:dlrgdn, :make], matchmissing = :notequal)
    isempty(r) && return missing
    out_names = vcat(
        names(r, Not(:uid, :dealer_geometry)),
        ["actual_distance", "shortest_distance", "normed_distance"],
    )
    empty_vec(name) = name in names(r) ? empty([], eltype(r[!, name])) : Float64[]
    init = DataFrame(out_names .=> empty_vec.(out_names))
    Folds.mapreduce(
        vcat,
        groupby(r, :uid),
        Folds.ThreadedEx(simd = true);
        init,
    ) do gdf
        dealers = gdf.dealer_geometry
        geometry = first(gdf.geometry)
        make = first(gdf.make)
        dist_info = normed_distance(dealers, geometry, make)
        @> gdf[[dist_info.i_shortest], :] begin
            hcat(DataFrame(; dist_info...))
            select(Not(:dealer_geometry, :i_shortest, :uid))
        end
    end
end

add_distance(r, t::Transaction) = add_distance(r, eachrow(State.df[dealer_idxs(t), :]))
