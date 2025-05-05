using Tyler
using WGLMakie
WGLMakie.activate!(resize_to=:body)
using ColorSchemes
using TileProviders
using Extents
using WellKnownGeometry
using Images
using ImageTransformations

import GeoFormatTypes as GFT
import GeoInterface as GI

to_maps_crs(x) = Geom.reproject!(AG.clone(x), detect_crs(x), maps_crs)

const CURRENT_MAP = Ref{Union{Tyler.Map,Nothing}}()

function init_map(window::Extent)
    rm(Tyler.CACHE_PATH[]) # force re-download, this is broken

    m = Tyler.Map(window, fetching_scheme=Tyler.SimpleTiling())

    wait(m, timeout=5)
    display(m)

    hidedecorations!(m.axis)

    return m
end
init_map(object) = init_map([object])
init_map(objects::AbstractArray) = init_map(mapreduce(GI.extent, Extents.union, objects))

function with_map(f!, window)
    m = init_map(window)
    CURRENT_MAP[] = m
    f!()

    @info "Adjust the map as you like and then type q to exit"
    while readline() != "q"
        sleep(0.01)
    end

    CURRENT_MAP[] = nothing
    return m
end

_getx(p::AG.IGeometry{AG.wkbPoint}) = AG.getx(p, 0)
_getx(mp::AG.IGeometry{AG.wkbMultiPoint}) = _getx(AG.getgeom(mp, 0))
_gety(p::AG.IGeometry{AG.wkbPoint}) = AG.gety(p, 0)
_gety(mp::AG.IGeometry{AG.wkbMultiPoint}) = _gety(AG.getgeom(mp, 0))

function image_scatter!(point, resolution, img_path; kwargs...)
    image = let i = Images.load(img_path)
        imrotate(i, π / 2, axes(i))
    end
    x = point |> to_maps_crs |> _getx
    y = point |> to_maps_crs |> _gety

    # resolution is in hundredths of the map extent
    Δx, Δy = let e = GI.extent(CURRENT_MAP[])
        xunit = (e.X[2] - e.X[1]) / 100
        yunit = (e.Y[2] - e.Y[1]) / 100
        resolution[1] * xunit / 2, resolution[2] * yunit / 2
    end

    image!((x - Δx, x + Δx), (y - Δy, y + Δy), image; kwargs...)
end
