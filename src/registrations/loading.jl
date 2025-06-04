using CSV
using DataFrames
using Cleaner
using PartialFunctions
using Dates
using Missings

import ..nonmissing

###### PARSING REGISTRATION CSVs

const CSV_OPTIONS = (
    maxwarnings = 1,
    quotechar = '\'',
    escapechar = '\'',
    stripwhitespace = true,
    stringtype = String,
    select = [
        :VIN,
        :REGEFFDATE,
        :VEHSALESPRICE,
        :VEHODMTRREADING,
        :OWNRZPCD,
        :DLRGDN,
        Symbol("LIENHLDRZPCD-1"),
        :JNKDATE,
        :TTLOWNRTYPECD,
    ],
    types = (_, name) ->
        get(Dict(:VIN => String, :VEHSALESPRICE => Float64), name, Union{String, Missing}),
)

try_parse_odometer(val) =
    if val == "EXEMPT FROM ODOMETER READING"
        missing
    else
        parse(Int64, replace(val, r"[^\d]" => ""))
    end

function transform_odometer(df)
    return transform(
        df, :vehodmtrreading => ByRow(passmissing(try_parse_odometer)), renamecols = false
    )
end

try_convert_zip(val) = lpad(val, 5, "0")

function transform_zip(df)
    return transform(
        df, names(df, r"zpcd") .=> ByRow(passmissing(try_convert_zip)), renamecols = false
    )
end

function try_parse_date(val)
    val = string(val)
    formats = [dateformat"yyyymmdd", dateformat"Y-m-d"]
    for format in formats
        try
            return Date(val, format)
        catch
            continue
        end
    end
    return error("Could not parse date: $val")
end

function transform_dates(df)
    return transform(
        df, names(df, r"date") .=> ByRow(passmissing(try_parse_date)), renamecols = false
    )
end

function process_chunk(chunk)
    @> chunk begin
        DataFrame
        polish_names_ROT
        transform_odometer
        transform_zip
        transform_dates
        subset(:regeffdate => ByRow(nonmissing(in(Date(2016, 1, 1):Date(2020, 1, 1)))))
    end
end

function load_registration_file_chunked(file)
    return (
        process_chunk(chunk)
        for chunk in CSV.Chunks(file; CSV_OPTIONS...)
    )
end
