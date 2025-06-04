import ..Geometry: gdf, to_coords

@memoize function zips()
    read_csv = @$ CSV.read(_, DataFrame)
    zip_demos = @> datadir("zip_demographics.csv") begin
        read_csv
        transform(:zip_code => ByRow(string), renamecols = false)
    end
    zip_centroids = @> datadir("zip_centroids.geojson") begin
        gdf
        transform(:geometry => ByRow(to_coords), renamecols = false)
    end

    return innerjoin(zip_demos, zip_centroids, on = :zip_code)
end

add_zips(r) = innerjoin(
    r,
    zips(),
    on = :ownrzpcd => :zip_code,
    matchmissing = :notequal,
)

filter_zips(r) = semijoin(
    r,
    zips(),
    on = :ownrzpcd => :zip_code,
    matchmissing = :notequal,
)
