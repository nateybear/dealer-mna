@memoize function dealership_count_map(ford_dealers::Vector{UUID})
    result = DefaultOrderedDict{UUID, Set{UUID}}(Set{UUID})
    for e in union(Dealers.Events, Date(2017):Quarter(1):Date(2020))
        Dealers.timetravel(e) do
            for (; gid, uid) in eachrow(Dealers.State.df)
                if uid in ford_dealers
                    push!(result[gid], uid)
                end
            end
        end
    end
    return result
end

@memoize single_operators(ford_dealers) =
    [gid for (gid, uids) in pairs(dealership_count_map(ford_dealers)) if length(uids) == 1]

const FRANCHISE_LOCK = ReentrantLock()

@memoize function franchise_graph_distances(t, make)
    network = lock(FRANCHISE_LOCK) do
        Dealers.fuzzytravel(t) do
            Dealers.compute_franchise_network!()
            return Dealers.State.franchise_network[make]
        end
    end

    graph = Graph()
    add_vertices!(graph, length(network.catchment_areas))
    for (i, j) in combinations(1:length(network.catchment_areas), 2)
        ca_i = network.catchment_areas[i]
        ca_j = network.catchment_areas[j]
        if AG.touches(ca_i, ca_j)
            add_edge!(graph, i, j)
        end
    end
    return network, bellman_ford_shortest_paths.(Ref(graph), 1:length(network.catchment_areas))
end

function hausman_price(mean_prices::GroupedDataFrame, t, j, k, f)
    mean_prices = mean_prices[mean_prices.keymap[(t, j)]] # pull out the relevant group immediately
    network, paths = franchise_graph_distances(t, first(j))
    i_f = findfirst(==(f), Dealers.State.df.uid)
    i_network = findfirst(==(i_f), network.dealers)
    if isnothing(i_network)
        @error "Dealer $f not found in franchise network for make $(first(j)) at time $t"
        return NaN
    end
    paths = paths[i_network]
    # start with dealers more than 4 hops away
    min_hops = 4
    i_distant_dealers = network.dealers[paths.dists .≥ min_hops]
    while length(i_distant_dealers) < 5 && (min_hops > 0)
        min_hops -= 1
        @warn "No distant dealers found for dealer $f at time $t for make $(j); reducing min_hops to $min_hops"
        i_distant_dealers = network.dealers[paths.dists .≥ min_hops]
    end
    if min_hops == 0
        @error "Could not find any distant dealers for dealer $f at time $t for make $(j)"
        return NaN
    end
    distant_dealers = Dealers.State.df.uid[i_distant_dealers]
    distant_prices = subset(
        mean_prices,
        :k => ByRow(==(k)),
        :f => ByRow(in(distant_dealers)),
        view = true,
    )
    if isempty(distant_prices)
        @error "No distant prices found for dealer $f at time $t for make $(j), trim $k, using price table re-inflation"
        price_table = Registrations.price_table()
        all_trims = subset(
            price_table,
            :make => ByRow(==(j[1])),
            :model => ByRow(==(j[2])),
            :model_year => ByRow(==(j[3])),
        )
        all_trims.j .= collect(zip(all_trims.make, all_trims.model, all_trims.model_year))
        my_relprice = pluck(all_trims, :trim_rank => ByRow(==(k))).trim_relprice
        distant_prices = subset(
            mean_prices,
            :f => ByRow(in(distant_dealers)),
            view = true,
        )
        if isempty(distant_prices)
            @error "Still no distant prices found for dealer $f at time $t for make $(j), giving up"
            return NaN
        end
        distant_prices = innerjoin(
            distant_prices,
            select(all_trims, :j, :trim_relprice, :trim_rank => :k),
            on = [:j, :k],
        )
        pull_mean((; μ, n)) = (; n, p = μ)
        transform!(distant_prices, :p => ByRow(pull_mean) => AsTable)
        distant_prices.p .= distant_prices.p .* my_relprice ./ distant_prices.trim_relprice
        return sum(distant_prices.p .* distant_prices.n) / sum(distant_prices.n)
    end
    return reduce(merge, distant_prices.p).μ
end

# If `row` is in scope, pull vals from row; otherwise use vals as variables directly
macro index(lookup, vals...)
    getval(v) = esc(:(@isdefined(row) ? row.$v : $v))
    index_vars = [Symbol(string(v) * "_index") for v in vals]
    assignments = [
        :($(index_vars[i]) = searchsortedfirst(
            collect(keys($(esc(lookup)).$(v))),
            $(getval(v)),
        )) for (i, v) in enumerate(vals)
    ]
    push_ops = [:(push!($(esc(lookup)).$(v)[$(getval(v))], i)) for v in vals]
    quote
        $(assignments...)
        i = CartesianIndex($(index_vars...))
        $(push_ops...)
        i
    end
end

function load_model_frames()
    r = @> read_arrow(snakemake.input["registrations"]) subset(
        :regeffdate => ByRow(>=(Date(2017))), view = true
    ) # events start in 2017 Q1
    ford_dealer_mask = Dealers.State.df.uid .∈ Ref(unique(r.dealer_uid))
    # Compute agent data, distances, firm product data, counts
    agent_data = Registrations.zips()
    agent_data.M = agent_data.population .÷ 48 # calibrate market size (2/HH, HH buys car every 6 years)
    transform!(agent_data, :median_earnings => ByRow(inv))
    demo_scales = Float32[std(agent_data[!, d]) for d in DEMOGRAPHIC_ATTRIBUTES]
    for (d, s) in zip(DEMOGRAPHIC_ATTRIBUTES, demo_scales)
        agent_data[!, d] ./= s
    end
    distances =
        miles.(agent_data.geometry, reshape(Dealers.State.df.geometry[ford_dealer_mask], 1, :)) # zips in rows, dealers in columns

    #### Done: distances
    distances = NamedArray(distances, (agent_data.zip_code, Dealers.State.df.uid[ford_dealer_mask])) # convert to miles

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

    r.quarter = Dates.quarter.(r.regeffdate)
    r.year = Dates.year.(r.regeffdate)
    r.t = tuple.(r.year, r.quarter) # (year, quarter)
    r.j = tuple.(r.make, r.model, r.model_year)

    body_classes = @> r groupby(:j) combine(:body_class => mode) eachrow Dict

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
        t, z = first(group.t), first(group.ownrzpcd)
        @info "Processing quarter $t, zip $z"
        Dealers.fuzzytravel(t) do
            group = DataFrames.rename(group, :dealer_uid => :f, :trim_rank => :k)
            group.g .=
                let joined = innerjoin(
                        group, Dealers.State.df[ford_dealer_mask, [:uid, :gid]], on = :f => :uid
                    )
                    replace(
                        joined.gid,
                        (
                            single_operators(Dealers.State.df.uid[ford_dealer_mask]) .=>
                                SINGLE_OPERATOR_GID
                        )...,
                    )
                end
            zt_counts = @> group groupby([:f, :j, :k]) combine(nrow)
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

    r = nothing
    GC.gc()
    pull_mean((; μ, n)) = (; n, p = μ)
    firm_product_data = @> mean_prices groupby([:t, :f, :j, :k, :g]) combine(
        :p => reduce$(merge) => :p
    )
    mean_prices = nothing
    GC.gc()
    firm_product_data.p_hausman = Folds.map(
        splat(hausman_price$(groupby(firm_product_data, [:t, :j]))),
        zip(
            firm_product_data.t,
            firm_product_data.j,
            firm_product_data.k,
            firm_product_data.f,
        ),
        ThreadedEx(simd = true),
    )
    subset!(firm_product_data, :p_hausman => ByRow(!isnan))
    transform!(firm_product_data, :p => ByRow(pull_mean) => AsTable)
    counts = semijoin(counts, firm_product_data, on = [:t, :f, :j, :k]) # we subsetted firm_product_data, match counts

    #### reshape dataframes into matrices and lookup tables ####
    nunique = nrow ∘ unique

    # fold trim rank into product key so demand arrays are (t, j_trim, f) and (z, t, j_trim, f)
    transform!(
        firm_product_data,
        [:j, :k] => ByRow((j, k) -> (j[1], j[2], j[3], Float32(k))) => :j_trim,
    )
    transform!(
        counts,
        [:j, :k] => ByRow((j, k) -> (j[1], j[2], j[3], Float32(k))) => :j_trim,
    )
    select!(firm_product_data, Not([:j, :k]))
    select!(counts, Not([:j, :k]))
    rename!(firm_product_data, :j_trim => :j)
    rename!(counts, :j_trim => :j)

    product_info_matrix = SparseArray{ProductInfo}(
        undef, nunique.(Ref(firm_product_data), (:t, :j, :f))
    )
    counts_matrix = SparseArray{Int16}(undef, nunique.(Ref(counts), (:z, :t, :j, :f)))

    transform!(counts, :nrow => ByRow(Int16), renamecols = false)
    disallowmissing!(counts)
    transform!(
        firm_product_data, :p => ByRow(Float32), :p_hausman => ByRow(Float32), renamecols = false
    )
    price_scale = Float32(std(firm_product_data.p))
    firm_product_data.p ./= price_scale
    firm_product_data.p_hausman ./= price_scale

    product_lookup = new_lookup(_ProductLookup, firm_product_data, :t, :j, :f)
    counts_lookup = new_lookup(CountsLookup, counts, :z, :t, :j, :f)

    @info "Filling product info matrix"
    for row in eachrow(firm_product_data)
        i = @index product_lookup t j f
        push!(product_lookup.make[first(row.j)], i) # add make after the fact
        push!(product_lookup.tjf[(row.t, row.j, row.f)], i) # add (t, j, f) after the fact
        body = get(body_classes, row.j[1:3], nothing)
        product_info_matrix[i] = (;
            is_coupe = body == "Coupe",
            is_suv = body == "SUV/Crossover",
            is_pickup = body == "Pickup",
            is_sedan = body == "Sedan",
            row.p,
            row.p_hausman,
            row.n,
            row.g,
        )
    end

    firm_product_data = nothing
    GC.gc()

    @info "Filling counts matrix"
    for row in eachrow(counts)
        i = @index counts_lookup z t j f
        counts_matrix[i] = row.nrow
    end

    @info "Creating ownership matrix"
    unique_t = collect(keys(product_lookup.t))
    unique_f = collect(keys(product_lookup.f))
    unique_g = setdiff(
        collect(keys(dealership_count_map(Dealers.State.df.uid[ford_dealer_mask]))) ∪
        [TRANSITION_GID, SINGLE_OPERATOR_GID],
        single_operators(Dealers.State.df.uid[ford_dealer_mask]),
    )
    ownership_matrix = SparseArray{Bool}(
        undef, (length(unique_t), length(unique_f), length(unique_g))
    )
    ownership_lookup = new_lookup(
        OwnershipLookup;
        t = unique_t,
        f = unique_f,
        g = unique_g,
    )
    for t in unique_t
        Dealers.fuzzytravel(t) do
            for (; uid, gid) in eachrow(Dealers.State.df)
                f, g = uid,
                (
                    if gid in single_operators(Dealers.State.df.uid[ford_dealer_mask])
                        SINGLE_OPERATOR_GID
                    else
                        gid
                    end
                )
                if f in unique_f && g in unique_g
                    i = @index ownership_lookup t f g
                    ownership_matrix[i] = true
                end
            end
        end
    end

    # @info "Creating dealer adjacency matrix"
    # unique_j = keys(product_lookup.j)
    # adjacency_matrix = SparseArray{Float32}(
    #     undef, (length(unique_t), length(unique_j), length(unique_f), length(unique_g))
    # )
    # adjacency_lookup = new_lookup(
    #     AdjacencyLookup;
    #     t = unique_t,
    #     j = unique_j,
    #     f = unique_f,
    #     g = unique_g,
    # )
    # sold_models(t, f, make) =
    #     Iterators.flatmap(keys(product_lookup.tjf)) do (t′, j, f′)
    #         if t′ == t && f′ == f && first(j) == make
    #             return [j]
    #         else
    #             return []
    #         end
    #     end
    # for t in unique_t
    #     Dealers.fuzzytravel(t) do
    #         Dealers.compute_franchise_network!()
    #         for (make, network) in pairs(Dealers.State.franchise_network)
    #             network_iter = zip(network.dealers, network.catchment_areas)
    #             for ((a_idx, a_geom), (b_idx, b_geom)) in
    #                 combinations(collect(network_iter), 2)
    #                 if AG.touches(a_geom, b_geom)
    #                     border = AG.intersection(a_geom, b_geom)
    #                     if AG.geomlength(border) == 0.0
    #                         @warn "Zero-length border between dealers $a_idx and $b_idx for make $make at time $t"
    #                         continue
    #                     end
    #                     a_fraction, b_fraction =
    #                         AG.geomlength(border) ./ AG.geomlength.([a_geom, b_geom])
    #                     f_a, f_b = Dealers.State.df.uid[[a_idx, b_idx]]
    #                     g_a, g_b = replace(
    #                         Dealers.State.df.gid[[a_idx, b_idx]],
    #                         (single_operators() .=> SINGLE_OPERATOR_GID)...,
    #                     )
    #                     for j in sold_models(t, f_a, make)
    #                         i = let f = f_a, g = g_b
    #                             @index adjacency_lookup t j f g
    #                         end
    #                         adjacency_matrix[i] += a_fraction
    #                     end
    #                     for j in sold_models(t, f_b, make)
    #                         i = let f = f_b, g = g_a
    #                             @index adjacency_lookup t j f g
    #                         end
    #                         adjacency_matrix[i] += b_fraction
    #                     end
    #                 end
    #             end
    #         end
    #     end
    # end

    # # apparently some empty groups; remove them
    nonempty_groups = findall(unique_g) do g
        !isempty(ownership_lookup.g[g]) #|| !isempty(adjacency_lookup.g[g])
    end

    @info "Found nonempty groups" length(nonempty_groups) length(unique_g)

    for obj in (product_lookup, counts_lookup, ownership_lookup)
        for i in propertynames(obj)
            for idxs in values(getproperty(obj, i))
                unique!(idxs)
            end
        end
    end

    return (
        distances = Float32.(distances),
        product_info = product_info_matrix,
        product_lookup = product_lookup,
        agent_data = agent_data,
        counts = counts_matrix,
        counts_lookup = counts_lookup,
        ownership = ownership_matrix[:, :, nonempty_groups],
        ownership_lookup = ownership_lookup,
        demo_scales = demo_scales,
        price_scale = price_scale,
        #adjacency = adjacency_matrix[:, :, :, nonempty_groups],
        #adjacency_lookup = adjacency_lookup,
    )
end
