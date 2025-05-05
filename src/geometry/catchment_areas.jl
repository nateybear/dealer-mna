@memoize function state_boundary_polygon()
    boundary = gdf(datadir("state_boundary.geojson")).geometry[1]
    boundary = AG.getgeom(AG.simplify(boundary, 2000), 0) # simplify to 2 km
    out = Tuple{Float64,Float64}[]
    for i in Base.OneTo(AG.ngeom(boundary))
        p = AG.getgeom(boundary, i - 1)
        push!(out, to_coords(p))
    end
    return reverse(out) # to make triangulation happy
end

@memoize state_boundary_polygon_sf() = state_boundary_polygon() |> AG.createpolygon

adjacent_dealers(p) = ((i, j) for (i, j) in combinations(eachindex(p), 2) if distance(p[i], p[j]) < 2mi)
adjacent_dealers(x, in_list) = (i for i in eachindex(in_list) if distance(x, in_list[i]) < 2mi)
has_adjacent_dealers(args...) = !isempty(adjacent_dealers(args...))
is_adjacent(x, y) = distance(x, y) < 2mi

function catchment_area(dealers)
    points = map(eachrow(dealers)) do (; latitude, longitude)
        assert_crs!((longitude, latitude))
    end
    points_sf = to_sf(points)

    # check if any points within 2 miles of each other
    too_close = iterate(adjacent_dealers(points))
    if !isnothing(too_close)
        ((i, j), _) = too_close
        error("Two dealers are too close to each other:\n$(dealers[i, :])\n\nand\n\n$(dealers[j, :])")
    end

    # Calculate the Voronoi tesselation using DelaunayTriangulation.jl
    bbox = let p = state_boundary_polygon()
        invoke.([first last], Tuple{Tuple}, p) |>
        extrema$(dims=1,) |>
        splat(Iterators.product) |>
        splat(vcat)
    end

    tessellation = points |>
                   triangulate |>
                   voronoi$(clip=true, clip_polygon=(bbox, (1, 2, 4, 3, 1)))

    # Convert to ArchGDAL geometries for use in GeoDataFrames
    geometry = let state = state_boundary_polygon_sf()
        points_with_idx = map(each_polygon(tessellation)) do p
            sf = @>> p begin
                map(get_polygon_point$tessellation)
                AG.createpolygon()
                AG.intersection(state)
                to_multigeom()
            end
            i = try
                only(findall(AG.contains(sf, p) for p in points_sf))
            catch
                @warn "Inexact tessellation, relying on distance" sf
                argmin(distance(p, sf) for p in points_sf)
            end
            (i, sf)
        end
        sort!(points_with_idx, by=first)
        map(last, points_with_idx)
    end

    hcat(
        dealers,
        DataFrame(; geometry)
    )
end
