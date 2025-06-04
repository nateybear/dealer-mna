import ..Dealers: Entry, Exit, Transaction
using Dates: Day, Quarter, Month

blackout_period(::Missing, ::Date) = missing
blackout_period(::Date, ::Missing) = missing
blackout_period(::Missing, ::Missing) = missing
blackout_period(from, to) = (from - BLACKOUT_LENGTH):(to + BLACKOUT_LENGTH)
blackout_period(dt) = blackout_period(dt, dt)
blackout_period(e::Entry) = blackout_period(e.first_regeffdate)
blackout_period(e::Exit) = blackout_period(e.last_regeffdate)
function blackout_period(t::Transaction)
    return blackout_period(t.last_seller_regeffdate, t.first_buyer_regeffdate)
end

function last_period_before(t::Transaction)
    return if ismissing(blackout_period(t))
        missing
    else
        (t.last_seller_regeffdate - BLACKOUT_LENGTH):(t.last_seller_regeffdate)
    end
end
function first_period_after(t::Transaction)
    return if ismissing(blackout_period(t))
        missing
    else
        (t.first_buyer_regeffdate):(t.first_buyer_regeffdate + BLACKOUT_LENGTH)
    end
end

# Julia days won't let you convert months/quarter to days, do it yourself
daylength(t::Day) = t.value
daylength(t::Month) = t.value * 30.25
daylength(t::Quarter) = t.value * 91
period_counter(d1) = d2 -> let
    diff_days = d2 - d1
    floor(Int, diff_days.value / daylength(PERIOD_LENGTH))
end
create_time_col(df, date) = @> df begin
    transform(:regeffdate => ByRow(period_counter(date)) => :T)
    subset(:T => ByRow(in(VALID_PERIODS)))
end
