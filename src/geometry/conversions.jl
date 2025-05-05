const GeomType = Union{AG.IGeometry{AG.wkbPoint},AG.IGeometry{AG.wkbPolygon}}
_multigeom_init(::AG.IGeometry{AG.wkbPoint}) = AG.createmultipoint()
_multigeom_init(::AG.IGeometry{AG.wkbPolygon}) = AG.createmultipolygon()
function to_multigeom(p::T) where {T<:GeomType}
    assert_crs!(p)
    out = _multigeom_init(T)
    AG.addgeom!(out, p)
    return out
end
to_multigeom(p::AG.IGeometry{AG.wkbMultiPoint}) = p
to_multigeom(p::AG.IGeometry{AG.wkbMultiPolygon}) = p

to_sf(points::AbstractVector) = to_sf.(points)
to_sf(p::Tuple{Float64,Float64}) = p |> AG.createpoint |> assert_crs!
to_coords(p::AG.IGeometry{AG.wkbPoint}) = (AG.getx(p, 0), AG.gety(p, 0))
to_coords(ps::AbstractVector{AG.IGeometry{AG.wkbPoint}}) = to_coords.(ps)
