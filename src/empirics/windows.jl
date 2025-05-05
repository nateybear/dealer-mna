import ..Dealers: Entry, Exit, Transaction

blackout_period(::Missing, ::Date) = missing
blackout_period(::Date, ::Missing) = missing
blackout_period(::Missing, ::Missing) = missing
blackout_period(from, to) = (from-BLACKOUT_LENGTH):(to+BLACKOUT_LENGTH)
blackout_period(dt) = blackout_period(dt, dt)
blackout_period(e::Entry) = blackout_period(e.first_regeffdate)
blackout_period(e::Exit) = blackout_period(e.last_regeffdate)
blackout_period(t::Transaction) = blackout_period(t.last_seller_regeffdate, t.first_buyer_regeffdate)

last_period_before(t::Transaction) = ismissing(blackout_period(t)) ?
                                     missing :
                                     (t.last_seller_regeffdate-BLACKOUT_LENGTH):t.last_seller_regeffdate
first_period_after(t::Transaction) = ismissing(blackout_period(t)) ?
                                     missing :
                                     t.first_buyer_regeffdate:(t.first_buyer_regeffdate+BLACKOUT_LENGTH)
