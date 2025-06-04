struct FieldDiff{F, T}
    i::Int64
    before::T
    after::T
end

# To apply is to travel backward in time
function apply!((; i, before)::FieldDiff{F}) where {F}
    getproperty(State.df, F)[i] = before
    return nothing
end

# To unapply is to travel forward in time
function unapply!((; i, after)::FieldDiff{F}) where {F}
    getproperty(State.df, F)[i] = after
    return nothing
end

# a full diff is just a tuple of field diffs
const Diff = Tuple{Vararg{FieldDiff}}
apply!(diff::Diff) = foreach(apply!, diff)
unapply!(diff::Diff) = foreach(unapply!, diff)

# apply this transaction, and return the diff
# N.B. the transaction might already be "applied" but we can fix that
function apply!(t::Transaction)
    @show t
    matched_gdn_buyer = Dict(
        gdn => findall(∋(gdn), State.df.gdns)
        for gdn in buyer_gdns(t)
    )
    matched_gdn_buyer = filter(!isempty ∘ last, matched_gdn_buyer)
    @show matched_gdn_buyer

    matched_gdn_seller = Dict(
        gdn => findall(∋(gdn), State.df.gdns)
        for gdn in seller_gdns(t)
    )
    matched_gdn_seller = filter(!isempty ∘ last, matched_gdn_seller)
    @show matched_gdn_seller

    if !isempty(matched_gdn_seller)
        @warn "Transaction appears to have already been applied, attempting to reverse" matched_gdn_seller
        if !isempty(matched_gdn_buyer)
            @warn "Both buyer and seller GDNs matched, possibly undefined behavior" matched_gdn_buyer matched_gdn_seller
        end
    end

    out = FieldDiff[]
    if !isempty(matched_gdn_buyer)
        # in this first case, try to handle where BOTH buyer and seller GDNs are present
        for (gdn, idxs) in matched_gdn_buyer
            m = subset(t.gdn_dba_map, :buyer_gdn => ByRow(==(gdn)))
            # need to modify gid, gdns, dba, entity
            gid = map(i -> FieldDiff{:gid, UUID}(i, t.seller_gid, t.buyer_gid), idxs)
            gdns = map(idxs) do i
                after_gdns = @> State.df.gdns[i] setdiff(seller_gdns(m))
                before_gdns = @> after_gdns begin
                    setdiff(buyer_gdns(m))
                    union(seller_gdns(m))
                end
                FieldDiff{:gdns, Set{String}}(i, before_gdns, after_gdns)
            end
            dbas = map(idxs) do i
                after_dbas = @> State.df.dbas[i] setdiff(seller_dbas(m))
                before_dbas = @> after_dbas begin
                    setdiff(buyer_dbas(m))
                    union(seller_dbas(m))
                end
                FieldDiff{:dbas, Set{Union{String, Missing}}}(i, before_dbas, after_dbas)
            end
            entity = map(idxs) do i
                FieldDiff{:entity, String}(i, t.seller_name, t.buyer_name)
            end
            append!(out, vcat(gid, gdns, dbas, entity))
        end
    end
    if !isempty(matched_gdn_seller)
        for (gdn, idxs) in matched_gdn_seller
            m = subset(t.gdn_dba_map, :seller_gdn => ByRow(==(gdn)))
            # need to modify gid, gdns, dba, entity
            gid = map(i -> FieldDiff{:gid, UUID}(i, t.buyer_gid, t.seller_gid), idxs)
            gdns = map(idxs) do i
                before_gdns = @> State.df.gdns[i] setdiff(buyer_gdns(m))
                after_gdns = @> before_gdns begin
                    setdiff(seller_gdns(m))
                    union(buyer_gdns(m))
                end
                FieldDiff{:gdns, Set{String}}(i, before_gdns, after_gdns)
            end
            dbas = map(idxs) do i
                before_dbas = @> State.df.dbas[i] setdiff(buyer_dbas(m))
                after_dbas = @> before_dbas begin
                    setdiff(seller_dbas(m))
                    union(buyer_dbas(m))
                end
                FieldDiff{:dbas, Set{Union{String, Missing}}}(i, before_dbas, after_dbas)
            end
            entity = map(idxs) do i
                FieldDiff{:entity, String}(i, t.buyer_name, t.seller_name)
            end
            append!(out, vcat(gid, gdns, dbas, entity))
        end
    end
    if isempty(matched_gdn_buyer) && isempty(matched_gdn_seller)
        @error "Both buyer and seller indexes are empty" matched_gdn_seller matched_gdn_buyer t
    end

    @show out
    out = Diff(out)

    apply!(out)
    return out
end

function apply!(e::Entry)
    idxs = findall(∋(e.gdn), State.df.gdns)
    @show e idxs

    isempty(idxs) && @error "No matching GDNs" idxs e

    out = Diff([FieldDiff{:active, Bool}(i, false, true) for i in idxs])
    @show out
    apply!(out)
    return out
end

function apply!(e::Exit)
    idxs = findall(∋(e.gdn), State.df.gdns)
    @show e idxs

    if isempty(idxs)
        @error "No matching GDNs" idxs e
        return Diff([])
    end

    out = Diff([FieldDiff{:active, Bool}(i, true, false) for i in idxs])
    @show out
    apply!(out)
    return out
end

# ASSUMPTION: order of Diffs array is the same order as Events array
Diffs = Diff[]

function compile_diffs!()
    sort!(Events, by = e -> e.when, rev = true) # most recent event is first
    diffs = map(apply!, Events)
    foreach(unapply!, reverse(diffs)) # back out from our diffs
    global Diffs = diffs
    return nothing
end

dealer_idxs(idx::Int64) = reduce(union, (fd.i for fd in Diffs[idx]))
dealer_idxs(e::AbstractEvent) = begin
    # get the index of the event in the Events array
    idx = findfirst(e2 -> e2.uid == e.uid, Events)
    return dealer_idxs(idx) # get the dealer indexes for that event
end

function timetravel(f, e::Union{AbstractEvent, DataFrameRow}; before = true)
    idx_last_event = if before
        searchsortedlast(Events, e, by = e -> e.when, rev = true)
    else
        searchsortedfirst(Events, e, by = e -> e.when, rev = true) - 1
    end

    for i in 1:idx_last_event
        apply!(Diffs[i])
    end

    try
        result = f() # try to execute the function
        for i in idx_last_event:-1:1
            unapply!(Diffs[i]) # unapply the diffs in reverse order
        end
        return result
    catch e
        for i in idx_last_event:-1:1
            unapply!(Diffs[i]) # unapply the diffs in reverse order
        end
        rethrow(e)
    end
end

function timetravel(f, d::Date)
    i = searchsortedlast([e.when for e in Events], d, rev = true)
    return timetravel(f, Events[i], before = true)
end
