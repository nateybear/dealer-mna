import ..Geometry: gdf, to_coords

@memoize function zips()
    zip_demos = @> CSV.read(datadir("zip_demographics.csv"), DataFrame) transform(:zip_code => ByRow(string), renamecols=false)
    zip_centroids = @> datadir("zip_centroids.geojson") gdf transform(:geometry => ByRow(to_coords), renamecols=false)

    return innerjoin(zip_demos, zip_centroids, on=:zip_code)
end

add_zips(r) = innerjoin(r, zips(), on=:ownrzpcd => :zip_code, matchmissing=:notequal)
