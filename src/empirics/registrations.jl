import ..Dealers: Entry, Exit, Transaction, add_distance, seller_gdns, buyer_gdns, timetravel, affected_gdns, State, dealer_idxs, Events
import ..Registrations: registrations, normalize_price, add_zips
import ..Geometry: distance
import ..snakemake
import ..nonmissing

import JLD2
using PartialFunctions

make_filter(::Nothing) = []
make_filter(make::String) = [:make => ByRow(==(make))]
make_filter(make::Vector{String}) = [:make => ByRow(in(make))]
make_filter(make::Set{String}) = make_filter(collect(make))

function last_registrations_before(t::Transaction; dealer_uid=nothing)
    ismissing(last_period_before(t)) && return missing

    out = timetravel(t) do
        i_extra = isnothing(dealer_uid) ? Set{Int}() : findfirst(==(dealer_uid), State.df.uid)
        gdns = isnothing(dealer_uid) ? seller_gdns(t) : State.df.gdns[i_extra]
        franchises = isnothing(dealer_uid) ? t.franchises : State.df.franchises[i_extra]
        @> registrations() begin
            subset(
                :regeffdate => ByRow(in(last_period_before(t))),
                :dlrgdn => ByRow(in(gdns)),
                make_filter(franchises)...
            )
            normalize_price
            add_distance
        end
    end

    (ismissing(out) || isempty(out)) && return missing

    out = create_time_col(out, t.last_seller_regeffdate + Day(1))

    # make sure registrations have all time periods
    # before the transaction, or else return missing
    before_periods = Set(p for p in VALID_PERIODS if p < 0)
    before_periods ⊆ out.T ? out : missing
end

function first_registrations_after(t::Transaction; dealer_uid=nothing)
    ismissing(first_period_after(t)) && return missing

    out = timetravel(t, before=false) do
        i_extra = isnothing(dealer_uid) ? Set{Int}() : findfirst(==(dealer_uid), State.df.uid)
        gdns = isnothing(dealer_uid) ? buyer_gdns(t) : State.df.gdns[i_extra]
        franchises = isnothing(dealer_uid) ? t.franchises : State.df.franchises[i_extra]
        @> registrations() begin
            subset(
                :regeffdate => ByRow(in(first_period_after(t))),
                :dlrgdn => ByRow(in(gdns)),
                make_filter(franchises)...
            )
            normalize_price
            add_distance
        end
    end

    (ismissing(out) || isempty(out)) && return missing
    out = create_time_col(out, t.first_buyer_regeffdate - Day(1))

    # make sure registrations have all time periods
    # after the transaction, or else return missing
    after_periods = Set(p for p in VALID_PERIODS if p ≥ 0)
    after_periods ⊆ out.T ? out : missing
end

function control_dealers(t; dealer_uid=nothing)
    timetravel(t) do
        # you are "tainted" if you have things happening to you, or you are closer than the radius
        tainted_gdns = reduce(union, (affected_gdns(t2) for t2 in Events if t2.uid != t.uid), init=Set{String}())

        # control dealers must be outside the radius of the treated dealerships and any extra dealers
        i_extra = isnothing(dealer_uid) ? Set{Int}() : findfirst(==(dealer_uid), State.df.uid)
        treated_dealerships = State.df[union(dealer_idxs(t), i_extra), :]

        #franchises are either the franchise of the extra dealer, or of the treated dealerships
        franchises = isnothing(dealer_uid) ? t.franchises : State.df.franchises[i_extra]
        intersects = !isempty ∘ intersect

        return subset(State.df,
            :geometry => ByRow(x -> all(>(RADIUS), distance(x, g) for g in treated_dealerships.geometry)),
            :gid => ByRow(!in([t.buyer_gid, t.seller_gid])),
            :gdns => ByRow(!in(tainted_gdns)),
            :franchises => ByRow(intersects$franchises)
        )
    end
end

function control_registrations_before(t; dealer_uid=nothing)
    gdns = reduce(union, control_dealers(t; dealer_uid).gdns)
    before_periods = Set(p for p in VALID_PERIODS if p < 0)
    i_extra = isnothing(dealer_uid) ? Set{Int}() : findfirst(==(dealer_uid), State.df.uid)
    franchises = isnothing(dealer_uid) ? t.franchises : State.df.franchises[i_extra]
    timetravel(t) do
        @> registrations() begin
            subset(
                :dlrgdn => ByRow(nonmissing(in(gdns))), # filter by the control group GDNS
                :regeffdate => ByRow(
                    # Only include registrations within the same period as the transaction
                    in(last_period_before(t))
                ),
                make_filter(franchises)...
            )
            normalize_price
            add_distance
            create_time_col(t.last_seller_regeffdate + Day(1))
            groupby(:dealer_uid)
            subset(:T => ⊇(before_periods))
        end
    end
end

function control_registrations_after(t; dealer_uid=nothing)
    gdns = reduce(union, control_dealers(t; dealer_uid).gdns)
    after_periods = Set(p for p in VALID_PERIODS if p ≥ 0)
    i_extra = isnothing(dealer_uid) ? Set{Int}() : findfirst(==(dealer_uid), State.df.uid)
    franchises = isnothing(dealer_uid) ? t.franchises : State.df.franchises[i_extra]
    timetravel(t, before=false) do
        @> registrations() begin
            subset(
                :dlrgdn => ByRow(nonmissing(in(gdns))), # filter by the control group GDNS
                :regeffdate => ByRow(
                    # Only include registrations within the same period as the transaction
                    in(first_period_after(t))
                ),
                make_filter(franchises)...
            )
            normalize_price
            add_distance
            create_time_col(t.first_buyer_regeffdate - Day(1))
            groupby(:dealer_uid)
            subset(:T => ⊇(after_periods))
        end
    end
end
