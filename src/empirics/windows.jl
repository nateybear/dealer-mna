import ..Dealers: AbstractEvent, eventtime
using Dates: Day, Quarter, Month

blackout_period(::Missing, ::Date) = missing
blackout_period(::Date, ::Missing) = missing
blackout_period(::Missing, ::Missing) = missing
blackout_period(from, to) = (from - BLACKOUT_LENGTH):(to + BLACKOUT_LENGTH - PERIOD_LENGTH)
blackout_period(dt) = blackout_period(dt, dt)
blackout_period(e::AbstractEvent) = blackout_period(eventtime(e))

function last_period_before(e::AbstractEvent)
    return if ismissing(blackout_period(e))
        missing
    else
        (eventtime(e) - BLACKOUT_LENGTH):(eventtime(e) - Day(1))
    end
end

function first_period_after(e::AbstractEvent)
    return if ismissing(blackout_period(e))
        missing
    else
        (eventtime(e)):(eventtime(e) + BLACKOUT_LENGTH - Day(1))
    end
end

create_eventtime_col!(df, ref_date::Date) = df.τ = eventtime.(df.t, Ref(ref_date))
