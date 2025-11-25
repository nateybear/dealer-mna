import ..Dealers: Entry, Exit, Transaction, add_distance, seller_gdns, buyer_gdns,
    fuzzytravel, affected_gdns, State, dealer_idxs, Events
import ..Registrations: registrations, manufacturer
import ..Geometry: distance, miles
import ..snakemake
import ..nonmissing

import JLD2
using PartialFunctions
import Folds
using FoldsThreads

make_filter(::Nothing) = []
make_filter(make::String) = [:make => ByRow(==(make))]
make_filter(make::Vector{String}) = [:make => ByRow(in(make))]
make_filter(make::Set{String}) = make_filter(collect(make))

function registrations_in_window(window; dealer_uids = [], franchises = [])
    return subset(
        registrations(),
        :dealer_uid => ByRow(in(dealer_uids)),
        :t => ByRow(in(window)),
        make_filter(franchises)...,
    )
end

function last_registrations_before(t; dealer_uid = nothing)
    all_registrations = treated_registrations(t; dealer_uid)
    ismissing(all_registrations) && return missing
    return subset(
        all_registrations,
        :τ => ByRow(<(0)),
    )
end

function first_registrations_after(t; dealer_uid = nothing)
    all_registrations = treated_registrations(t; dealer_uid)
    ismissing(all_registrations) && return missing
    return subset(
        all_registrations,
        :τ => ByRow(>=(0)),
    )
end

function treated_registrations(t; dealer_uid = nothing)
    window = blackout_period(t)
    ismissing(window) && return missing

    i_extra = if isnothing(dealer_uid)
        Set{Int}()
    else
        findfirst(==(dealer_uid), State.df.uid)
    end

    franchises = isnothing(dealer_uid) ? t.franchises : State.df.franchises[i_extra]
    if !isempty(franchises ∩ t.franchises)
        intersect!(franchises, t.franchises) # if this is a dealer that sells same makes, only look at those makes
    end

    dealer_uids = isnothing(dealer_uid) ? State.df.uid[dealer_idxs(t)] : Set([dealer_uid])

    out = registrations_in_window(window; dealer_uids, franchises)
    create_eventtime_col!(out, eventtime(t))

    return isempty(out) ? missing : out
end

function control_dealers(t; dealer_uid = nothing, dealer_demographics)
    # control dealers must be outside the radius of the treated dealerships and any extra dealers
    i_extra = if isnothing(dealer_uid)
        Set{Int}()
    else
        findfirst(==(dealer_uid), State.df.uid)
    end

    #franchises are either the franchise of the extra dealer, or of the treated dealerships
    franchises = isnothing(dealer_uid) ? t.franchises : State.df.franchises[i_extra]
    if !isempty(franchises ∩ t.franchises)
        intersect!(franchises, t.franchises) # if this is a dealer that sells same makes, only look at those makes
    end

    intersects = !isempty ∘ intersect

    # you are "tainted" if you have things happening to you, or you are closer than the radius
    # of an event within its blackout period and a franchise that we care about
    tainted_uids = Folds.mapreduce(
        union,
        Events,
        WorkStealingEx(simd = true),
        init = Set{UUID}(),
    ) do e
        e.when > Date(2020) && return Set{UUID}()
        radioactive = dealer_idxs(e) ∪ i_extra
        too_close = Set{Int}()
        blackout = blackout_period(e)
        if !ismissing(blackout) && eventtime(t) in blackout
            for (i, (g, f)) in enumerate(zip(State.df.geometry, State.df.franchises))
                if i in radioactive
                    continue
                end
                if intersects(manufacturer.(f), manufacturer.(franchises))
                    dynamic_radius = let uids = State.df.uid[radioactive]
                        map(uids) do u
                            min(
                                pluck(
                                    dealer_demographics, :uid => ByRow(==(u))
                                ).closest_third,
                                miles(RADIUS),
                            )
                        end
                    end
                    if any(
                        miles.(Ref(g), State.df.geometry[radioactive]) .≤ dynamic_radius
                    )
                        push!(too_close, i)
                    end
                end
            end
        end
        return State.df.uid[radioactive ∪ too_close]
    end

    return subset(
        State.df,
        :gid => ByRow(!in([t.buyer_gid, t.seller_gid])),
        :uid => ByRow(!in(tainted_uids)),
        :franchises => ByRow(intersects$franchises),
        :active => ByRow(identity),
    )
end

# iterate a series of events and maybe dealer uids,
# and compile a unique list of control dealer uids, windows, and franchises
function iter_control_dealers(ts::AbstractVector; dealer_demographics)
    out = DefaultDict{UUID, Set{Tuple{Date, Set{String}}}}(Set{Tuple{Date, Set{String}}})
    for t in ts
        t, dealer_uid = t isa Tuple ? t : (t, nothing)

        i_extra = if isnothing(dealer_uid)
            Set{Int}()
        else
            findfirst(==(dealer_uid), State.df.uid)
        end

        #franchises are either the franchise of the extra dealer, or of the treated dealerships
        franchises = isnothing(dealer_uid) ? t.franchises : State.df.franchises[i_extra]
        if !isempty(franchises ∩ t.franchises)
            intersect!(franchises, t.franchises) # if this is a dealer that sells same makes, only look at those makes
        end

        ismissing(blackout_period(t)) && continue
        controls = control_dealers(t; dealer_uid, dealer_demographics)
        for uid in controls.uid
            windows = out[uid]
            for window in eventtime.(eventtime(t), VALID_PERIODS)
                push!(windows, (window, franchises))
            end
        end
    end
    reshaped = DataFrame(
        dealer_uids = collect(keys(out)),
        windows = collect(values(out)),
    )
    transform!(reshaped, :windows => ByRow((map$first) ∘ collect) => :window)
    transform!(
        reshaped, :windows => ByRow((map$last) ∘ collect) => :franchises
    )
    select!(reshaped, Not(:windows))
    reshaped = @> reshaped flatten(:window) flatten(:franchises)
    # regroup into as wide of window/dealer uid combos as possible
    reshaped = @> reshaped groupby([:dealer_uids, :franchises]) combine(
        :window => Set, renamecols = false
    )
    reshaped = @> reshaped groupby([:franchises, :window]) combine(
        :dealer_uids => Set, renamecols = false
    )

    return eachrow(reshaped)
end
