###### KEEP EVERYTHING IN CRS NAD83 / Texas Centric Albers Equal Area, data is assumed to be lat/lon

tx_crs = GFT.EPSG(3083)
latlon_crs = GFT.EPSG(4326)
maps_crs = GFT.EPSG(3857)

@enum CRS begin
    Texas
    LatLon
    GoogleMaps # really, any map service
end

### UNSAFE: reproject mutates the geometry
reproject!(geom, from_crs, to_crs=tx_crs) = AG.reproject(geom, from_crs, to_crs, order=:trad)

@memoize austin_latlon() = AG.createpoint(-97.7311762, 30.2873681)
@memoize austin_tx() = reproject!(AG.clone(austin_latlon()), latlon_crs)
@memoize austin_mapscrs() = reproject!(AG.clone(austin_latlon()), latlon_crs, maps_crs)

detect_crs(geom::AG.IGeometry) =
    let
        which_closest = argmin(AG.distance$geom, (austin_latlon(), austin_tx(), austin_mapscrs()))
        if which_closest === austin_latlon()
            Val(LatLon)
        elseif which_closest === austin_tx()
            Val(Texas)
        else
            Val(GoogleMaps)
        end
    end

assert_crs!(geom::AG.IGeometry) = assert_crs!(geom, detect_crs(geom))
assert_crs!(geom, ::Val{Texas}) = geom
assert_crs!(geom, ::Val{LatLon}) = reproject!(geom, latlon_crs)
assert_crs!(geom, ::Val{GoogleMaps}) = reproject!(geom, maps_crs)
assert_crs!(point::Tuple{Float64,Float64}) =
    let geom = AG.createpoint(point)
        assert_crs!(geom)
        to_coords(geom)
    end
