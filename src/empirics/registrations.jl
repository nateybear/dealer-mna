import ..Dealers: Entry, Exit, Transaction, add_distance, seller_gdns, buyer_gdns, timetravel, affected_gdns, State, dealer_idxs
import ..Registrations: registrations, normalize_price, add_zips
import ..Geometry: distance
import ..snakemake
import ..nonmissing

import JLD2

make_filter(::Nothing) = []
make_filter(make::String) = [:make => ByRow(==(make))]
make_filter(make::Vector{String}) = [:make => ByRow(in(make))]
make_filter(make::Set{String}) = make_filter(collect(make))

function last_registrations_before(t::Transaction, make=nothing)
    ismissing(last_period_before(t)) && return missing

    out = timetravel(t) do
        @> registrations() begin
            subset(
                :regeffdate => ByRow(in(last_period_before(t))),
                :dlrgdn => ByRow(in(seller_gdns(t))),
                make_filter(isnothing(make) ? t.franchises : make)...
            )
            normalize_price
            add_distance
        end
    end

    ismissing(out) || isempty(out) && return missing

    out = create_time_col(out, t.last_seller_regeffdate + Day(1))

    # make sure registrations have all time periods
    # before the transaction, or else return missing
    before_periods = Set(p for p in VALID_PERIODS if p < 0)
    before_periods ⊆ out.T ? out : missing
end

function first_registrations_after(t::Transaction, make=nothing)
    ismissing(first_period_after(t)) && return missing

    out = timetravel(t, before=false) do
        @> registrations() begin
            subset(
                :regeffdate => ByRow(in(first_period_after(t))),
                :dlrgdn => ByRow(in(buyer_gdns(t))),
                make_filter(isnothing(make) ? t.franchises : make)...
            )
            normalize_price
            add_distance
        end
    end

    ismissing(out) || isempty(out) && return missing
    out = create_time_col(out, t.first_buyer_regeffdate - Day(1))

    # make sure registrations have all time periods
    # after the transaction, or else return missing
    after_periods = Set(p for p in VALID_PERIODS if p ≥ 0)
    after_periods ⊆ out.T ? out : missing
end

function control_dealers(t)
    transactions_for_analysis = JLD2.load_object(snakemake.input["transactions_for_analysis"])
    timetravel(t) do
        # you are "tainted" if you are undergoing a transaction that overlaps with the blackout period of another transaction
        tainted_gdns = reduce(union, (affected_gdns(t2) for t2 in transactions_for_analysis if t2.uid != t.uid), init=Set{String}())
        treated_dealerships = State.df[dealer_idxs(t), :]

        return subset(State.df,
            :geometry => ByRow(x -> all(>(RADIUS), distance(x, g) for g in treated_dealerships.geometry)),
            :gid => ByRow(!in([t.buyer_gid, t.seller_gid])),
            :gdns => ByRow(!in(tainted_gdns)),
            :franchises => ByRow(
                # Only include dealerships that have at least one make in common with the treated dealerships
                x -> !isempty(intersect(t.franchises, x))
            )
        )
    end
end

function control_registrations_before(t)
    gdns = reduce(union, control_dealers(t).gdns)
    before_periods = Set(p for p in VALID_PERIODS if p < 0)
    timetravel(t) do
        @> registrations() begin
            subset(
                :dlrgdn => ByRow(nonmissing(in(gdns))), # filter by the control group GDNS
                :regeffdate => ByRow(
                    # Only include registrations within the same period as the transaction
                    in(last_period_before(t))
                ),
                make_filter(t.franchises)...
            )
            normalize_price
            add_distance
            create_time_col(t.last_seller_regeffdate + Day(1))
            groupby(:dealer_uid)
            subset(:T => ⊇(before_periods))
        end
    end
end

function control_registrations_after(t)
    gdns = reduce(union, control_dealers(t).gdns)
    after_periods = Set(p for p in VALID_PERIODS if p ≥ 0)
    timetravel(t, before=false) do
        @> registrations() begin
            subset(
                :dlrgdn => ByRow(nonmissing(in(gdns))), # filter by the control group GDNS
                :regeffdate => ByRow(
                    # Only include registrations within the same period as the transaction
                    in(first_period_after(t))
                ),
                make_filter(t.franchises)...
            )
            normalize_price
            add_distance
            create_time_col(t.first_buyer_regeffdate - Day(1))
            groupby(:dealer_uid)
            subset(:T => ⊇(after_periods))
        end
    end
end
