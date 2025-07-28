# Standard libraries
using Dates

# Data processing libraries
using CSV
using DataFrames
using Arrow
using JLD2
using Missings
using XLSX

# Utilities and extensions
using Cleaner: polish_names_ROT
using DrWatson
using Lazy: @>
using Memoize
using PartialFunctions
import Unicode
import JSON3

import ..snakemake, ..here

@memoize all_registrations() = DataFrame(
    Arrow.Table(
        get(
            snakemake.input,
            "unmatched_registrations",
            "data/arrow/unmatched_registrations.arrow",
        ),
    ),
)

function normed(y)
    let f = (ByRow ∘ passmissing)() do x
            @> x Unicode.normalize strip replace(r"\s+" => " ", r"[^\w\s]" => "") uppercase
        end
        f(y)
    end
end

# Types for the dealer group graph
include("graphs.jl")
function add_groups(dealers, col = :business_name)
    Base.include(Main, srcdir("dealers/graphs.jl")) # our JLD2 file was saved with a different version of the graph
    graph = JLD2.load_object(datadir("dealer_groups_graph.jld2"))
    l = collect(labels(graph))
    group_ids = DataFrame(business_name = l, group_id = [graph[a].id for a in l])
    return innerjoin(dealers, group_ids, on = col => :business_name)
end

@memoize all_makes() = unique(Arrow.Table(snakemake.input["popular_models"]).make)
const make_aliases = [
    (r"CDJR?F?", "CHRYSLER"),
    (r"CDJR?F?", "DODGE"),
    (r"CDJR?F?", "JEEP"),
    (r"CDJRF?", "RAM"),
    ("CHEVY", "CHEVROLET"),
    ("DODGE", "RAM"),
    ("GM", "CHEVROLET"),
    ("GM", "GMC"),
    ("MERCEDES", "MERCEDES-BENZ"),
    ("BENZ", "MERCEDES-BENZ"),
    ("VW", "VOLKSWAGEN"),
]
const false_positives = DefaultDict(
    Vector{String},
    "RAM" => ["RAMTUFF AUTO GROUP", "RAM AUTO SALES"],
    "FORD" => ["BOBBY FORD CHRYSLER DODGE JEEP RAM"],
    "HONDA" => ["HONDA OF ORANGE", "HONDA OF HOUSTON"], # motorcycle dealers
)
const true_negatives = open(JSON3.read, here("src/dealers/true_negatives.json"))

function add_franchises(
    dealers, cols = [:dbaname, :business_name]; gdn_col = :license_number
)
    @> dealers transform(
        [gdn_col, cols...] =>
            ByRow() do gdn, cols...
                search_string = uppercase(coalesce(cols...))
                if haskey(true_negatives, search_string)
                    dealer = true_negatives[Symbol(search_string)]
                    if dealer isa AbstractVector{String}
                        return Set(dealer)
                    else
                        return Set(get(dealer, Symbol(gdn), String[]))
                    end
                end

                dealer_franchises = Set{String}()

                for make in all_makes()
                    if occursin(r"\b" * make * r"\b", search_string) &&
                        !(search_string in false_positives[make])
                        push!(dealer_franchises, make)
                    end
                end
                for alias in make_aliases
                    if occursin(r"\b" * alias[1] * r"\b", search_string)
                        push!(dealer_franchises, alias[2])
                    end
                end
                dealer_franchises
            end => :franchises,
    )
end

const unique_dealer_fields = [:license_number, :business_name, :dbaname, :physical_address]
function load_clean_dealers()
    out = @> datadir("Dealers/Motor Vehicle Dealer-2022-May-19-04-00-02.xlsx") begin
        XLSX.readtable(1, infer_eltypes = true)
        DataFrame
        polish_names_ROT
        subset(:license_type => ByRow(==("Motor Vehicle")))
        transform(
            :zip => ByRow(passmissing(x -> String(strip(x)[1:5]))), renamecols = false
        )
        transform(
            [:physical_address, :phys_address_two, :city, :state, :zip] .=>
                ByRow(passmissing(titlecase)),
            renamecols = false,
        )
        transform([:dbaname, :business_name] .=> normed)
        unique(unique_dealer_fields)
    end

    matching_reg = semijoin(
        out, all_registrations(), on = :license_number => :dlrgdn, matchmissing = :notequal
    )
    missing_reg = antijoin(
        out, all_registrations(), on = :license_number => :dlrgdn, matchmissing = :notequal
    )

    # dealerships might have opened/changed hands after my registration data
    return vcat(
        matching_reg,
        subset(missing_reg, :active_date => ByRow(x -> last(x, 4) >= "2020")),
    )
end

@memoize function transactions()
    out = @> datadir("Dealers/Mergers_Entry_Exit.xlsx") begin
        XLSX.readtable(1, infer_eltypes = true)
        DataFrame
        polish_names_ROT
        subset(:gdntype => ByRow(==("Motor Vehicle")))
        select(
            :app_name,
            :transaction_dt,
            r"gdn$",
            r"business_name$",
            :dbaname => :buyer_dbaname,
            :seller_dbaname,
        )
    end

    transform!(out,
        names(out, r"dbaname") .=> normed,
        names(out, r"business_name") .=> normed,
    )

    unique!(out)

    with_seller_info = subset(out, :seller_gdn => ByRow(!ismissing))
    without_seller_info = @> out begin
        subset(:seller_gdn => ByRow(ismissing))
        select(:app_name, :transaction_dt, r"^buyer")
        innerjoin(
            repaired_legacy_transactions(),
            on = [:app_name],
            matchmissing = :notequal,
        )
    end

    out = vcat(with_seller_info, without_seller_info, cols = :union)

    unique_sellers = unique(out.seller_gdn)
    unique_buyers = unique(out.buyer_gdn)

    seller_regeff_dates = @> DataFrame(seller_gdn = unique_sellers) begin
        innerjoin(
            select(all_registrations(), :dlrgdn => :seller_gdn, :regeffdate),
            on = :seller_gdn,
            matchmissing = :notequal,
        )
        dropmissing(:regeffdate)
        groupby(:seller_gdn)
        combine(:regeffdate => maximum => :last_seller_regeffdate)
    end

    buyer_regeff_dates = @> DataFrame(buyer_gdn = unique_buyers) begin
        innerjoin(
            select(all_registrations(), :dlrgdn => :buyer_gdn, :regeffdate),
            on = :buyer_gdn,
            matchmissing = :notequal,
        )
        dropmissing(:regeffdate)
        groupby(:buyer_gdn)
        combine(:regeffdate => minimum => :first_buyer_regeffdate)
    end

    @> out begin
        leftjoin(seller_regeff_dates, on = :seller_gdn, matchmissing = :notequal)
        leftjoin(buyer_regeff_dates, on = :buyer_gdn, matchmissing = :notequal)
        select(
            :app_name,
            :transaction_dt,
            r"gdn$",
            r"business_name",
            r"dbaname",
            r"regeffdate$",
        )
    end
end

@memoize function entrants()
    out = @> datadir("Dealers/Mergers_Entry_Exit.xlsx") begin
        XLSX.readtable(2, infer_eltypes = true)
        DataFrame
        polish_names_ROT
        subset(:gdntype => ByRow(==("Motor Vehicle")))
        select(
            :approved_dt,
            :application_reason_c => :reason,
            :app_name,
            :associated_gdn => :gdn,
            :dba_on_gdn => :dbaname,
            r"business_name",
        )
        transform(:approved_dt => ByRow(Date), renamecols = false)
        unique
        antijoin(transactions(), on = :app_name, matchmissing = :notequal)
        antijoin(transactions(), on = :gdn => :buyer_gdn) # if you're created in the buying process, ignore you
        semijoin(all_registrations(), on = :gdn => :dlrgdn, matchmissing = :notequal)
    end

    transform!(out, [:dbaname, names(out, r"business_name")...] .=> normed)

    regeff_dates = @> DataFrame(gdn = unique(out.gdn)) begin
        innerjoin(
            select(all_registrations(), :dlrgdn => :gdn, :regeffdate),
            on = :gdn,
            matchmissing = :notequal,
        )
        dropmissing(:regeffdate)
        groupby(:gdn)
        combine(:regeffdate => minimum => :first_regeffdate)
    end

    @> out leftjoin(regeff_dates, on = :gdn, matchmissing = :notequal)
end

@memoize function exits()
    out = @> datadir("Dealers/Mergers_Entry_Exit.xlsx") begin
        XLSX.readtable(3, infer_eltypes = true)
        DataFrame
        polish_names_ROT
        subset(:gdntype => ByRow(==("Motor Vehicle")))
        select(
            :gdnclosed_dt,
            :gdnclose_reason => :reason,
            :gdnnumber => :gdn,
            :dbaname,
            :business_name,
        )
        transform([:dbaname, :business_name] .=> normed)
        unique
        antijoin(transactions(), on = :gdn => :seller_gdn, matchmissing = :notequal) # exclude exits as part of a sale
        semijoin(all_registrations(), on = :gdn => :dlrgdn, matchmissing = :notequal)
    end

    regeff_dates = @> DataFrame(gdn = unique(out.gdn)) begin
        innerjoin(
            select(all_registrations(), :dlrgdn => :gdn, :regeffdate),
            on = :gdn,
            matchmissing = :notequal,
        )
        dropmissing(:regeffdate)
        groupby(:gdn)
        combine(:regeffdate => maximum => :last_regeffdate)
    end

    @> out leftjoin(regeff_dates, on = :gdn, matchmissing = :notequal)
end

"""
Some transaction info was saved after a database change in comments, so try and extract that
"""
@memoize function repaired_legacy_transactions()
    legacy = @> datadir("Dealers/Legacy_Comment_Info.xlsx") XLSX.readtable(
        1, infer_eltypes = true
    ) DataFrame polish_names_ROT
    transform!(
        legacy,
        :comments_seller_business_name =>
            ByRow() do x
                license_numbers_with_offset = map(eachmatch(r"[A-Z]\d{4,}", x)) do y
                    (String(y.match), y.offset)
                end
                seller_gdn = filter(startswith("P"), first.(license_numbers_with_offset))
                license_number_offset = minimum(last.(license_numbers_with_offset))
                seller_business_name = first(x, license_number_offset - 1)
                seller_business_name = replace(seller_business_name, r"[-\s]+$" => "")
                (; seller_gdn, seller_business_name)
            end => AsTable,
    )
    @> legacy select(:app_name, r"^seller") flatten(:seller_gdn)
end

@memoize geocoded_dealers() = CSV.read(
    datadir("geocoded_dealers.csv"),
    DataFrame,
    types = Dict(:zip => String, :geo_zip => String),
)

function add_geocoding(dealers)
    @> dealers begin
        innerjoin(
            geocoded_dealers(),
            on = [:physical_address, :phys_address_two, :city, :state, :zip],
            matchmissing = :equal,
            validate = (false, true),
        )
        rename(:lon => :longitude, :lat => :latitude) # my bad...
        transform(
            [:longitude, :latitude] =>
                ByRow(passmissing(Geometry.assert_crs! ∘ AG.createpoint)) => :geometry,
        )
    end
end

@memoize msas() = @> datadir("cbsas.geojson") Geometry.gdf subset(:lsad => ByRow(==("M1")))

# nothing means not in an msa, missing means we don't have lat/lon geocoded
function add_msas(dealers)
    @> dealers begin
        transform(
            :geometry =>
                (ByRow ∘ passmissing)() do g
                    dealer_msas = subset(msas(), :geometry => ByRow(AG.within$g))
                    isempty(dealer_msas) && return nothing
                    msa = first(dealer_msas).name
                    @> msa split("-") first replace(", TX" => "")
                end => :msa,
        )
    end
end

function clean_loading_caches!()
    let memoized_funcs = (
            transactions,
            entrants,
            exits,
            repaired_legacy_transactions,
            geocoded_dealers,
        )
        foreach(empty! ∘ memoize_cache, memoized_funcs)
    end
end
