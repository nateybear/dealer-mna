#### MSA type
const MSA = Union{String, Missing, Nothing}

using Random
_rng = Xoshiro(78704)

# Dealership represents a single LOCATION
@kwdef struct Dealership
    uid::UUID = uuid4(_rng)
    active::Bool = true
    gid::UUID
    gdns::Set{String}
    dbas::Set{Union{String, Missing}}
    entity::String
    address::String
    geometry::Tuple{Float64, Float64}
    msa::MSA = nothing
    franchises::Set{String}
end

struct FranchiseNetwork{N, G <: AG.IGeometry}
    dealers::SVector{N, Int64}
    catchment_areas::SVector{N, G}
end

@kwdef mutable struct _State
    df::Union{DataFrame, StructVector{Dealership}} = StructVector(Dealership[])
    when::Date = today()
    franchise_network::Dict{String, FranchiseNetwork} = Dict{String, FranchiseNetwork}()
    dealer_groups::Dict{UUID, Set{Int64}} = Dict{UUID, Set{Int64}}()
end

State = _State()

abstract type AbstractEvent end

@kwdef struct Transaction <: AbstractEvent
    uid::UUID = uuid4(_rng)
    buyer_name::String
    seller_name::String
    buyer_gid::UUID
    seller_gid::UUID
    when::Date
    app_name::String
    franchises::Set{String}
    # ↓ map of buyer gdns/dbas to seller gdns/dbas (remember, we roll backwards)
    gdn_dba_map::DataFrame
    first_buyer_regeffdate::Union{Date, Missing} = missing
    last_seller_regeffdate::Union{Date, Missing} = missing
end

seller_gdns(t::Transaction) = Set(t.gdn_dba_map.seller_gdn)
seller_gdns(d::DataFrame) = Set(d.seller_gdn)
buyer_gdns(t::Transaction) = Set(t.gdn_dba_map.buyer_gdn)
buyer_gdns(d::DataFrame) = Set(d.buyer_gdn)
seller_dbas(t::Transaction) = Set(t.gdn_dba_map.seller_dba)
seller_dbas(d::DataFrame) = Set(d.seller_dba)
buyer_dbas(t::Transaction) = Set(t.gdn_dba_map.buyer_dba)
buyer_dbas(d::DataFrame) = Set(d.buyer_dba)
entities(t::Transaction) = Set([t.buyer_name, t.seller_name])

@kwdef struct Entry <: AbstractEvent
    uid::UUID = uuid4(_rng)
    reason::String = ""
    when::Date
    app_name::String
    business_name::String
    gid::UUID
    franchises::Set{String}
    gdn::String
    dba::Union{String, Missing}
    first_regeffdate::Union{Date, Missing} = missing
end

@kwdef struct Exit <: AbstractEvent
    uid::UUID = uuid4(_rng)
    reason::String = ""
    when::Date
    business_name::String
    gid::UUID
    franchises::Set{String}
    gdn::String
    dba::Union{String, Missing}
    last_regeffdate::Union{Date, Missing} = missing
end

Events = AbstractEvent[]

affected_gdns(e::Union{Entry, Exit}) = Set([e.gdn])
affected_gdns(t::Transaction) = seller_gdns(t) ∪ buyer_gdns(t)
entities(e::Union{Entry, Exit}) = Set([e.business_name])
regeffdate(e::Entry) = e.first_regeffdate
regeffdate(e::Exit) = e.last_regeffdate
regeffdate(t::Transaction) = mid(t.first_buyer_regeffdate, t.last_seller_regeffdate)

mid(_, _) = missing
mid(d1::Date, d2::Date) = d1 + (d2 - d1) ÷ 2
