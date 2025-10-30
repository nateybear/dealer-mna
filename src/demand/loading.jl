@memoize function single_operators()
    dealership_count_map = DefaultOrderedDict{UUID, Set{UUID}}(Set{UUID})
    for e in Dealers.Events
        Dealers.timetravel(e) do
            for (; gid, uid) in eachrow(Dealers.State.df)
                push!(dealership_count_map[gid], uid)
            end
        end
    end
    return [gid for (gid, uids) in pairs(dealership_count_map) if length(uids) == 1]
end

# see usage below...
macro index(lookup, vals...)
    # Generate variable assignments for each dimension
    assignments = [:($(Symbol(string(val) * "_index")) = searchsortedfirst(collect(keys($(esc(lookup)).$(val))), $(esc(:row)).$(val))) for val in vals]
    
    # Generate the CartesianIndex construction
    index_vars = [Symbol(string(val) * "_index") for val in vals]
    cartesian_construction = :(i = CartesianIndex($(index_vars...)))
    
    # Generate push! operations for each lookup dimension
    push_operations = [:(push!($(esc(lookup)).$(val)[$(esc(:row)).$(val)], i)) for val in vals]
    
    quote
        $(assignments...)
        $cartesian_construction
        $(push_operations...)
        i
    end
end

function load_model_frames()
    #### COMPUTE AGENT DATA, DISTANCES, FIRM PRODUCT DATA, COUNTS ####
    agent_data = Registrations.zips()
    agent_data.M = agent_data.population .÷ 48 # calibrate market size for now ( 2/ HH, HH buys car every six years)
    transform!(agent_data, :median_earnings => ByRow(inv))
    distances = miles.(agent_data.geometry, reshape(Dealers.State.df.geometry, 1, :)) # zips in rows, dealers in columns

    #### Done: distances
    distances = NamedArray(distances, (agent_data.zip_code, Dealers.State.df.uid)) # convert to miles

    #### Done: agent data
    agent_data = Dict{String, AgentInfo}(
        row.zip_code => (
            median_earnings_inv = row.median_earnings_inv,
            urban_pct = row.urban_pct,
            black_pct = row.black_pct,
            hispanic_pct = row.hispanic_pct,
            M = row.M,
        ) for row in eachrow(agent_data)
    )

    r = @> Registrations.registrations() subset(
        :regeffdate => ByRow(>=(Date(2017))), view = true
    ) # events start in 2017 Q1
    r.quarter = Dates.quarter.(r.regeffdate)
    r.year = Dates.year.(r.regeffdate)
    r.t = tuple.(r.year, r.quarter) # (year, quarter)
    r.j = tuple.(r.make, r.model, r.model_year)

    mean_prices = DataFrame(
        t = Tuple{Int64, Int64}[],
        f = UUID[],
        j = Tuple{String, String, Int64}[],
        k = Float64[],
        g = UUID[],
        p = Mean{Float64, EqualWeight}[],
    )
    counts = DataFrame(
        t = Tuple{Int64, Int64}[],
        f = Union{UUID, Missing}[],
        j = Union{Tuple{String, String, Int64}, Missing}[],
        k = Union{Float64, Missing}[],
        z = String[],
        nrow = Int64[],
    )

    fitmean(v) = fit!(Mean(), v)
    for group in groupby(r, [:t, :ownrzpcd])
        t = first(group.t)
        z = first(group.ownrzpcd)
        total = 0
        @info "Processing quarter $t, zip $z"
        Dealers.fuzzytravel(t) do
            group = DataFrames.rename(group, :dealer_uid => :f, :trim_rank => :k)
            group.g .=
                let joined = innerjoin(
                        group,
                        Dealers.State.df[:, [:uid, :gid]],
                        on = :f => :uid,
                    )
                    replace(joined.gid, (single_operators() .=> SINGLE_OPERATOR_GID)...)
                end
            zt_counts = @> group groupby([:f, :j, :k]) combine(nrow)
            total += sum(zt_counts.nrow)
            zt_counts.z .= z
            zt_counts.t .= Ref(t)
            append!(counts, zt_counts)
            zt_prices = @> group groupby([:f, :j, :k, :g]) combine(
                :vehsalesprice => fitmean => :p
            )
            zt_prices.t .= Ref(t)
            append!(mean_prices, zt_prices)
        end
    end

    pull_mean((; μ, n)) = (; n, p = μ)
    firm_product_data = @> mean_prices groupby([:t, :f, :j, :k, :g]) combine(
        :p => pull_mean ∘ reduce$(merge) => AsTable
    )
    empty!(Memoize.memoize_cache(Registrations.registrations))


    #### reshape dataframes into matrices and lookup tables ####
    nunique = nrow ∘ unique

    product_info_matrix = SparseArray{ProductInfo}(
        undef, nunique.(Ref(firm_product_data), (:t, :j, :k, :f))
    )
    counts_matrix = SparseArray{Int16}(undef, nunique.(Ref(counts), (:z, :t, :j, :k, :f)))

    transform!(counts, :nrow => ByRow(Int16), :k => ByRow(Float32), renamecols = false)
    transform!(firm_product_data, :p => ByRow(Float32), :k => ByRow(Float32), renamecols = false)

    sort!(counts, [:z, :t, :j, :k, :f])
    sort!(firm_product_data, [:t, :j, :k, :f])

    product_lookup = new_product_lookup(firm_product_data)
    counts_lookup = new_counts_lookup(counts)

    @info "Filling product info matrix"
    for row in eachrow(firm_product_data)
        i = @index product_lookup t j k f
        push!(product_lookup.make[first(row.j)], i) # add make after the fact
        push!(product_lookup.tjf[(row.t, row.f, row.j)], i) # add (t, f, j) after the fact
        domestic = first(row.j) in DOMESTIC_MAKES
        product_info_matrix[i] = (;
            domestic,
            row.p,
            row.n,
            row.g,
        )
    end

    @info "Filling counts matrix"
    for row in eachrow(counts)
        i = @index counts_lookup z t j k f
        counts_matrix[i] = row.nrow
    end

    return (
        distances = Float32.(distances),
        product_info = product_info_matrix,
        product_lookup = product_lookup,
        agent_data = agent_data,
        counts = counts_matrix,
        counts_lookup = counts_lookup,
    )
end
