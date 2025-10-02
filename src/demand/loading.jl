# Apply f to the dealer state as of the end of quarter t, but fuzz the effective dates of transactions by ±1 quarter.
# Also, consolidate all single-operator dealers into a single "virtual" dealer with gid single_operator_gid.
# Finally, for any dealer involved in a transaction during quarter t (after fuzzing), set its gid to transition_gid.
function fuzzytravel(f, t::Tuple{Int64, Int64})
    year, quarter = t
    start_date = Date(year) + Quarter(quarter)
    end_date = lastdayofquarter(start_date)
    date_range = start_date:end_date

    fuzzed_events = findall(eachindex(Dealers.Events)) do i
        e = Dealers.Events[i]
        isempty(Dealers.Diffs[i]) && return false
        d = Dealers.regeffdate(e)
        ismissing(d) && return false
        fuzz_dates = (d - Quarter(1)) : (d + Quarter(1))
        return !isempty(intersect(fuzz_dates, date_range))
    end

    i_before = isempty(fuzzed_events) ? findfirst(e -> e.when > start_date, Dealers.Events) : minimum(fuzzed_events)

    Dealers.timetravel(Dealers.Events[i_before]) do
        df = deepcopy(Dealers.State.df)
        df.gid[df.gid .∈ Ref(single_operators())] .= SINGLE_OPERATOR_GID
        for i in fuzzed_events
            df.gid[i] = TRANSITION_GID
            gdn_diffs = [ d for d in Dealers.Diffs[i] if d isa Dealers.FieldDiff{:gdns} ]
            for d in gdn_diffs
                union!(df.gdns[d.i], d.after) # combine all gdns in fuzzy events to make sure we get all the registrations
            end
        end
        return f(df)
    end
end

@memoize function single_operators()
    dealership_count_map = DefaultDict{UUID, Set{UUID}}(Set{UUID})
    for e in Dealers.Events
        Dealers.timetravel(e) do
            for (; gid, uid) in eachrow(Dealers.State.df)
                push!(dealership_count_map[gid], uid)
            end
        end
    end
    return [ gid for (gid, uids) in pairs(dealership_count_map) if length(uids) == 1 ]
end

function load_model_frames()
    agent_data = Registrations.zips()
    agent_data.M = agent_data.population .÷ 48 # calibrate market size for now ( 2/ HH, HH buys car every six years)
    distances = distance.(agent_data.geometry, reshape(Dealers.State.df.geometry, 1, :)) # zips in rows, dealers in columns
    distances = NamedArray(miles.(distances), (agent_data.zip_code, Dealers.State.df.uid)) # convert to miles
    r = Registrations.registrations()
    r.quarter = Dates.quarter.(r.regeffdate)
    r.year = Dates.year.(r.regeffdate)
    r.t = tuple.(r.year, r.quarter) # (year, quarter)
    r.j = tuple.(r.make, r.model, r.model_year)
    mean_prices = DataFrame(
        t = Tuple{Int64, Int64}[],
        f = UUID[],
        j = Tuple{String, String, Int64}[],
        k = Float64[],
        make = String[],
        g = UUID[],
        p = Mean{Float64, EqualWeight}[],
    )
    counts = DataFrame(
        t = Tuple{Int64, Int64}[],
        f = UUID[],
        j = Tuple{String, String, Int64}[],
        k = Float64[],
        z = String[],
        nrow = Int64[],
    )
    
    fitmean = @$ fit!(Mean(), _)
    for group in groupby(r, [:t, :ownrzpcd])
        t = first(group.t)
        z = first(group.ownrzpcd)
        @info "Processing quarter $t, zip $z"
        group = Registrations.normalize_price(group)
        DataFrames.rename!(group, :trim_rank => :k)
        fuzzytravel(t; single_operators, single_operator_gid, transition_gid) do state
            group = Dealers.add_distance(group; state)
            ismissing(group) && return
            DataFrames.rename!(group, :dealer_uid => :f, :dealer_gid => :g)
            zt_counts = @> group groupby([:f, :j, :k]) combine(nrow)
            zt_counts.z .= z
            zt_counts.t .= Ref(t)
            append!(counts, zt_counts)
            zt_prices = @> group groupby([:f, :j, :k, :make, :g]) combine(:vehsalesprice => fitmean => :p)
            zt_prices.t .= Ref(t)
            append!(mean_prices, zt_prices)
        end
    end

    product_data = @> mean_prices groupby([:t, :f, :j, :k, :make, :g]) combine(:p => (ps -> reduce(merge, ps).μ) => :p)

    return (; distances, product_data, agent_data, counts)
end
