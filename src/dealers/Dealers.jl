module Dealers

import ArchGDAL as AG
using DataFrames
using DataStructures: DefaultDict
using Dates
using DrWatson
using JLD2
using Lazy: @>
using Memoize
using PartialFunctions
using StructArrays
using StaticArrays
using UUIDs

import ..Geometry

include("loading.jl")
include("types.jl")
include("diffs.jl")
include("distances.jl")

const DEALER_DATE_RANGE = Date(2017, 1, 1):Date(2022, 5, 19)

function init_dealers!()
    State.when = Date(2022, 5, 19) # date of the dealers file
    State.df = StructVector(Dealership[])

    dealers = @> load_clean_dealers() begin
        add_franchises
        subset(:franchises => ByRow(!isempty))
        add_geocoding
        add_msas
        add_groups
    end

    for d in eachrow(dealers)
        dbas = Set([coalesce(d.dbaname_normed, d.business_name_normed)])
        gdns = Set([d.license_number])
        franchises = d.franchises

        # conditions for merging:
        # 1. same address
        address = "$(d.physical_address), $(d.city) $(d.zip)"
        idx_same_address = findfirst(==(address), State.df.address)
        if !isnothing(idx_same_address)
            old = State.df[idx_same_address]
            @info "Merging by address" old new = NamedTuple(d)
            State.df.dbas[idx_same_address] = union(old.dbas, dbas)
            State.df.gdns[idx_same_address] = union(old.gdns, gdns)
            State.df.franchises[idx_same_address] = union(old.franchises, franchises)
            continue
        end

        Geometry.assert_crs!(d.geometry)
        geometry = Geometry.to_coords(d.geometry)

        # 2. same license number and adjacent location (within 5 miles)
        idx_same_gdns = findfirst(State.df) do d2
            d.license_number ∈ d2.gdns && Geometry.is_adjacent(geometry, d2.geometry)
        end
        if !isnothing(idx_same_gdns)
            # license number is the same
            old = State.df[idx_same_gdns]
            @info "Merging by adjacency" old new = NamedTuple(d) distance = Geometry.distance(
                geometry, old.geometry
            )
            State.df.dbas[idx_same_gdns] = union(old.dbas, dbas)
            State.df.franchises[idx_same_gdns] = union(old.franchises, franchises)
            continue
        end

        # 3. same dba and adjacent location
        idx_same_dba = findfirst(State.df) do d2
            !isempty(d2.dbas ∩ dbas) && Geometry.is_adjacent(geometry, d2.geometry)
        end
        if !isnothing(idx_same_dba)
            old = State.df[idx_same_dba]
            @info "Merging by dba" old new = NamedTuple(d) distance = Geometry.distance(
                geometry, old.geometry
            )
            State.df.dbas[idx_same_dba] = union(old.dbas, dbas)
            State.df.franchises[idx_same_dba] = union(old.franchises, franchises)
            continue
        end

        # 4. same group id, same franchise, and adjacent location
        # (looking at you Nissan of Brownsville)
        idx_same_group = findfirst(State.df) do d2
            d.group_id == d2.gid &&
                d2.franchises == franchises &&
                Geometry.is_adjacent(geometry, d2.geometry)
        end
        if !isnothing(idx_same_group)
            old = State.df[idx_same_group]
            @info "Merging by group" old new = NamedTuple(d) distance = Geometry.distance(
                geometry, old.geometry
            )
            State.df.dbas[idx_same_group] = union(old.dbas, dbas)
            State.df.gdns[idx_same_group] = union(old.gdns, gdns)
            continue
        end

        # else this is a new dealership
        @debug "New dealership" NamedTuple(d)
        push!(
            State.df,
            Dealership(;
                dbas,
                address,
                geometry,
                d.msa,
                gdns,
                franchises,
                gid = d.group_id,
                entity = d.business_name_normed,
            ),
        )
    end
end

# make this a DataFrame composed of StaticVectors (SVector), which is a performance optimization
function freeze_dealers!()
    N = length(State.df)
    State.df = @> State.df begin
        collect
        SVector{N}
        StructVector
        DataFrame
    end
    return nothing
end

function compute_franchise_network!()
    all_franchises = collect(reduce(union, State.df.franchises))
    State.franchise_network = Dict(
        begin
            dealer_idx = findall(∋(franchise), State.df.franchises)
            geoms_df = DataFrame(
                idx = dealer_idx,
                longitude = first.(State.df.geometry[dealer_idx]),
                latitude = last.(State.df.geometry[dealer_idx]),
            )
            network = Geometry.catchment_area(geoms_df)
            N = nrow(network)

            franchise => FranchiseNetwork(
                SVector{N}(network.idx), SVector{N}(network.geometry)
            )
        end
        for franchise in all_franchises
    )
    return nothing
end

function compute_group_map!()
    State.dealer_groups =
        let out = DefaultDict{UUID, Set{Int64}}(Set{Int64})
            for (i, d) in enumerate(eachrow(State.df))
                out[d.gid] = union(out[d.gid], i)
            end
            Dict(out)
        end
    return nothing
end

#### transactions

function init_transactions!()
    franchise_transactions = @> transactions() begin
        #in date range
        transform(:transaction_dt => ByRow(Date), renamecols = false)
        subset(:transaction_dt => ByRow(in(DEALER_DATE_RANGE)))
        dropmissing(r"business_name$")

        # add seller group
        add_groups(:seller_business_name)
        rename(:group_id => :seller_group_id)

        # add buyer group
        add_groups(:buyer_business_name)
        rename(:group_id => :buyer_group_id)

        # add franchises
        add_franchises(
            [:buyer_dbaname, :seller_dbaname, :buyer_business_name, :seller_business_name],
            gdn_col = :buyer_gdn,
        )
        subset(:franchises => ByRow(!isempty))

        add_regeffdates(maximum, :last_seller_regeffdate, :seller_gdn)
        add_regeffdates(minimum, :first_buyer_regeffdate, :buyer_gdn)

        # return a grouped df
        groupby(:app_name)
    end

    onlyunique = only ∘ unique

    for t in franchise_transactions
        buyer_name = onlyunique(t.buyer_business_name_normed)
        seller_name = onlyunique(t.seller_business_name_normed)
        buyer_gid = onlyunique(t.buyer_group_id)
        seller_gid = onlyunique(t.seller_group_id)
        app_name = onlyunique(t.app_name)
        when = onlyunique(t.transaction_dt)
        franchises = reduce(union, t.franchises)

        # this is a relation with no guarantees of a function
        gdn_dba_map = @> t begin
            select(
                :buyer_gdn,
                :seller_gdn,
                :buyer_dbaname_normed => :buyer_dba,
                :seller_dbaname_normed => :seller_dba,
            )
            unique
        end

        let seller_gdn_mapping = @> gdn_dba_map begin
                select(:buyer_gdn, :seller_gdn)
                unique
                groupby(:seller_gdn)
            end
            if !all(==(1) ∘ nrow, seller_gdn_mapping)
                @warn "Indeterminate transaction mapping" app_name gdn_dba_map
            end
        end

        first_buyer_regeffdate = emptymissing(minimum)(
            skipmissing(t.first_buyer_regeffdate)
        )
        last_seller_regeffdate = emptymissing(maximum)(
            skipmissing(t.last_seller_regeffdate)
        )

        # if another in the same week between the same parties, merge
        i = findfirst(Events) do e
            abs(e.when - when) < Week(1) &&
                e isa Transaction &&
                e.buyer_name == buyer_name &&
                e.seller_name == seller_name
        end

        if isnothing(i)
            push!(
                Events,
                Transaction(;
                    buyer_name,
                    seller_name,
                    buyer_gid,
                    seller_gid,
                    app_name,
                    gdn_dba_map,
                    first_buyer_regeffdate,
                    last_seller_regeffdate,
                    when,
                    franchises,
                ),
            )
        else
            # merge the transactions
            old = Events[i]
            @info "Merging transactions" old new = t
            Events[i] = Transaction(;
                buyer_name,
                seller_name,
                buyer_gid,
                seller_gid,
                app_name,
                gdn_dba_map = unique(vcat(old.gdn_dba_map, gdn_dba_map)),
                first_buyer_regeffdate = (
                                             old.first_buyer_regeffdate,
                                             first_buyer_regeffdate,
                                         ) |>
                                         skipmissing |>
                                         emptymissing(minimum),
                last_seller_regeffdate = (
                                             old.last_seller_regeffdate,
                                             last_seller_regeffdate,
                                         ) |>
                                         skipmissing |>
                                         emptymissing(maximum),
                when,
                franchises = union(old.franchises, franchises),
            )
        end
    end
end

function Base.setdiff(df1::DataFrame, df2::DataFrame)
    if names(df1) == names(df2)
        antijoin(df1, df2, on = names(df1), matchmissing = :equal)
    else
        error("DataFrames have different column names")
    end
end

function init_entry_exit!()
    franchise_entrants = @> entrants() begin
        add_franchises(gdn_col = :gdn)
        add_regeffdates(minimum, :first_regeffdate)
        add_groups
        subset(
            :franchises => ByRow(!isempty),
            :approved_dt => ByRow(in(DEALER_DATE_RANGE)),
        )
    end
    franchise_exits = @> exits() begin
        add_franchises(gdn_col = :gdn)
        add_regeffdates(maximum, :last_regeffdate)
        add_groups
        subset(
            :franchises => ByRow(!isempty),
            :gdnclosed_dt => ByRow(in(DEALER_DATE_RANGE)),
        )
    end

    # special cases of entry
    relocations = subset(
        franchise_entrants,
        :reason => ByRow(contains(r"(moving)|(relocation)"i)),
    )
    franchise_entrants = Base.setdiff(franchise_entrants, relocations)

    sunsetted_gdns = subset(relocations, [:approved_dt, :first_regeffdate] => ByRow(<)) # most "relocations" actually just used their old GDNs. These have to be matched back to a row in franchise_exit
    matched_sunsetted_exits = semijoin(
        franchise_exits,
        sunsetted_gdns,
        on = [:business_name_normed, :franchises],
    ) # this match is okay (have to check by hand)
    franchise_exits = Base.setdiff(franchise_exits, matched_sunsetted_exits)

    # these entity changes kept same ownership and GDN, so just remove them as entries and exits
    entrants_entity_change = subset(
        franchise_entrants,
        :reason => ByRow(contains(r"entity change"i)),
    )
    matched_exits_entity_change = semijoin(
        franchise_exits,
        entrants_entity_change,
        on = :dbaname_normed,
        matchmissing = :notequal,
    )
    exits_entity_change = subset(
        franchise_exits,
        :reason => ByRow(contains(r"entity change"i)),
    )
    franchise_entrants = Base.setdiff(franchise_entrants, entrants_entity_change)
    franchise_exits = @> franchise_exits begin
        Base.setdiff(matched_exits_entity_change)
        Base.setdiff(exits_entity_change)
    end

    # what's left are actual entries and exits
    for e in eachrow(franchise_entrants)
        push!(Events,
            Entry(;
                e.reason, e.app_name, e.first_regeffdate, e.gdn, e.franchises,
                when = e.approved_dt,
                business_name = e.business_name_normed,
                gid = e.group_id,
                dba = e.dbaname_normed,
            ))
    end

    for e in eachrow(franchise_exits)
        push!(Events,
            Exit(;
                e.reason, e.franchises, e.gdn, e.last_regeffdate,
                when = e.gdnclosed_dt,
                business_name = e.business_name_normed,
                gid = e.group_id,
                dba = e.dbaname_normed,
            ))
    end

    # TODO: repair duplicate license numbers (there's so few that I'm skipping for now)
    #
    # for relocations with a new number issued, run through the events and apply the new license numbers
    #= gdn_repair = innerjoin(
        select(sunsetted_gdns, :gdn => :old_gdn, :business_name_normed, :franchises),
        select(matched_sunsetted_exits, :gdn => :new_gdn, :business_name_normed, :franchises),
        on=[:business_name_normed, :franchises]
    )
    for (i, e) in Events

    end =#
end

function init!()
    @info "Initializing dealers"
    init_dealers!()
    @info "Initializing transactions"
    init_transactions!()
    @info "Initializing entry/exit"
    init_entry_exit!()
    @info "Freezing dealers"
    freeze_dealers!()
    @info "Compiling diffs"
    compile_diffs!()
    clean_loading_caches!()
    return nothing
end

function write_to_disk!(filename)
    @info "Writing dealers to disk"
    # Save the dealers data
    @save filename State Events Diffs
    return nothing
end

function load_from_disk!(filename)
    @info "Loading dealers from disk"
    global State, Events, Diffs
    # Load the dealers data
    @load filename State Events Diffs
    return nothing
end

end
