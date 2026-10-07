@memoize function dealership_count_map(ford_dealers::Vector{UUID})
    result = OrderedDict{UUID, Set{UUID}}()
    for e in union(Dealers.Events, Date(2017):Quarter(1):Date(2020))
        Dealers.timetravel(e) do
            for (; gid, uid) in eachrow(Dealers.State.df)
                if uid in ford_dealers
                    get!(result, gid, Set{UUID}())
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
            Dealers.compute_franchise_network!([make])
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

function load_model_frames()
    kept_triples = Set(
        (row.model, row.model_year, row.extra) for
        row in eachrow(read_arrow(snakemake.input["kept_trims"]))
    )
    r = @> read_arrow(snakemake.input["registrations"]) subset(
        :regeffdate => ByRow(>=(Date(2017))),
        [:model, :model_year, :extra] =>
            ByRow((m, y, e) -> (m, y, e) ∈ kept_triples),
        view = true,
    ) # events start in 2017 Q1; kept_trims restricts to (model, model_year, extra) with ≥ AVG_Q_THRESHOLD avg dealer-quarter sales
    ford_dealer_mask = Dealers.State.df.uid .∈ Ref(unique(r.dealer_uid))

    # Agent data: zip-level demographics + market size; standardize demographics by std-dev.
    agent_data_df = Registrations.zips()
    agent_data_df.M = agent_data_df.population .÷ 48 # 2/HH, HH buys car every 6 years
    transform!(agent_data_df, :median_earnings => ByRow(inv))
    demo_scales = Float32[std(agent_data_df[!, d]) for d in DEMOGRAPHIC_ATTRIBUTES]
    for (d, s) in zip(DEMOGRAPHIC_ATTRIBUTES, demo_scales)
        agent_data_df[!, d] ./= s
    end

    # Distances: zips × dealers in miles, then standardized
    distances =
        miles.(
            agent_data_df.geometry,
            reshape(Dealers.State.df.geometry[ford_dealer_mask], 1, :),
        )
    distances = NamedArray(
        Float32.(distances),
        (agent_data_df.zip_code, Dealers.State.df.uid[ford_dealer_mask]),
    )
    dist_scale = Float32(std(distances))
    distances ./= dist_scale

    # agent_data: Dict from zip_code → demographics + M
    agent_data = Dict{String, AgentInfo}(
        row.zip_code => (
            median_earnings_inv = row.median_earnings_inv,
            urban_pct = row.urban_pct,
            black_pct = row.black_pct,
            hispanic_pct = row.hispanic_pct,
            M = row.M,
        ) for row in eachrow(agent_data_df)
    )

    # zips: canonical column order — same order Registrations.zips() returns
    zips = String.(agent_data_df.zip_code)

    r.quarter = Dates.quarter.(r.regeffdate)
    r.year = Dates.year.(r.regeffdate)
    r.t = tuple.(r.year, r.quarter)
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
    firm_product_data.p_hausman = map(
        splat(hausman_price$(groupby(firm_product_data, [:t, :j]))),
        zip(
            firm_product_data.t,
            firm_product_data.j,
            firm_product_data.k,
            firm_product_data.f,
        ),
    )
    subset!(firm_product_data, :p_hausman => ByRow(!isnan))
    transform!(firm_product_data, :p => ByRow(pull_mean) => AsTable)
    counts = semijoin(counts, firm_product_data, on = [:t, :f, :j, :k])

    # Fold trim rank into product key: (make, model, model_year, trim_rank) is the canonical j.
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

    transform!(counts, :nrow => ByRow(Int32), renamecols = false)
    disallowmissing!(counts)
    transform!(
        firm_product_data, :p => ByRow(Float32), :p_hausman => ByRow(Float32), renamecols = false
    )
    price_scale = Float32(std(firm_product_data.p))
    firm_product_data.p ./= price_scale
    firm_product_data.p_hausman ./= price_scale

    # Deterministic δ ordering: sort by (t, j, f), then dedup. A given (t, j, f) can appear
    # in multiple rows when a dealer's group affiliation (g) transitions mid-quarter — those
    # rows describe the same product. The old loader silently collapsed them by overwriting
    # product_info on each CartesianIndex assignment, leaving the LAST row's values. Mirror
    # that: after sorting, drop all but the last row of each consecutive (t, j, f) run.
    sort!(firm_product_data, [:t, :j, :f])
    n_before_dedup = nrow(firm_product_data)
    keep = trues(n_before_dedup)
    @inbounds for i in 1:(n_before_dedup - 1)
        if firm_product_data.t[i] == firm_product_data.t[i + 1] &&
            firm_product_data.j[i] == firm_product_data.j[i + 1] &&
            firm_product_data.f[i] == firm_product_data.f[i + 1]
            keep[i] = false
        end
    end
    firm_product_data = firm_product_data[keep, :]
    N = nrow(firm_product_data)
    if N != n_before_dedup
        @info "deduplicated firm_product_data" n_before = n_before_dedup n_after = N n_dropped =
            n_before_dedup - N
    end

    @info "Building flat product_info (N=$N)"
    product_info = Dict{Int32, ProductInfo}()
    sizehint!(product_info, N)
    for (i, row) in enumerate(eachrow(firm_product_data))
        body = get(body_classes, row.j[1:3], nothing)
        product_info[Int32(i)] = (;
            is_coupe = body == "Coupe",
            is_suv = body == "SUV/Crossover",
            is_pickup = body == "Pickup",
            is_sedan = body == "Sedan",
            row.p,
            row.p_hausman,
            n = Int16(row.n),
            row.g,
        )
    end

    # Build ProductLookup with flat Int32 indices.
    product_lookup = ProductLookup(
        OrderedDict{QuarterKey, Vector{Int32}}(),
        OrderedDict{VehicleKey, Vector{Int32}}(),
        OrderedDict{UUID, Vector{Int32}}(),
        OrderedDict{String, Vector{Int32}}(),
        OrderedDict{TJFKey, Int32}(),
    )
    for i in 1:N
        row = firm_product_data[i, :]
        t_key = row.t::QuarterKey
        j_key = row.j::VehicleKey
        f_key = row.f::UUID
        make_key = first(j_key)
        push!(get!(product_lookup.t, t_key, Int32[]), Int32(i))
        push!(get!(product_lookup.j, j_key, Int32[]), Int32(i))
        push!(get!(product_lookup.f, f_key, Int32[]), Int32(i))
        push!(get!(product_lookup.make, make_key, Int32[]), Int32(i))
        product_lookup.tjf[(t_key, j_key, f_key)] = Int32(i)
    end

    firm_product_data = nothing
    GC.gc()

    # Per-block layout: one block per (year, quarter), sorted.
    block_keys = sort(collect(keys(product_lookup.t)))
    n_blocks = length(block_keys)
    block_param_idxs = [sort(product_lookup.t[k]) for k in block_keys]

    K = length(PRODUCT_ATTRIBUTES)
    Dn = length(DEMOGRAPHIC_ATTRIBUTES)
    Z = length(zips)

    # tjf_to_block_k: flat δ_idx → (bi, k_local)
    tjf_to_block_k = Dict{Int32, Tuple{Int32, Int32}}()
    for (bi, param_idxs) in enumerate(block_param_idxs)
        for (k_local, p) in enumerate(param_idxs)
            tjf_to_block_k[p] = (Int32(bi), Int32(k_local))
        end
    end

    # X_blocks[bi]: (K+1) × N_bi — product attribute flags + standardized price.
    @info "Building X_blocks"
    X_blocks = Vector{Matrix{Float32}}(undef, n_blocks)
    for bi in 1:n_blocks
        param_idxs = block_param_idxs[bi]
        N_bi = length(param_idxs)
        M = Matrix{Float32}(undef, K + 1, N_bi)
        for (k_local, p) in enumerate(param_idxs)
            info = product_info[p]
            for (pi, attr) in enumerate(PRODUCT_ATTRIBUTES)
                M[pi, k_local] = Float32(getfield(info, attr))
            end
            M[K + 1, k_local] = info.p
        end
        X_blocks[bi] = M
    end

    # dist_blocks[bi]: N_bi × Z — gather distances[zip, dealer_uid] per product.
    @info "Building dist_blocks"
    dealer_uids = names(distances, 2)
    dealer_to_col = Dict{UUID, Int}(u => i for (i, u) in enumerate(dealer_uids))
    zip_to_row = Dict{String, Int}(z => i for (i, z) in enumerate(names(distances, 1)))
    # dist_zf: Z × F dense, in our zip order
    F = length(dealer_uids)
    dist_zf = Matrix{Float32}(undef, Z, F)
    for (zi, z) in enumerate(zips)
        zrow = zip_to_row[z]
        for f in 1:F
            dist_zf[zi, f] = Float32(distances.array[zrow, f])
        end
    end
    # tjf_keys in 1:N order (sorted)
    tjf_keys = collect(keys(product_lookup.tjf))   # OrderedDict iteration order = insertion order
    # Validate: keys were inserted in 1:N order, so position == index
    dist_blocks = Vector{Matrix{Float32}}(undef, n_blocks)
    for bi in 1:n_blocks
        param_idxs = block_param_idxs[bi]
        N_bi = length(param_idxs)
        M = Matrix{Float32}(undef, N_bi, Z)
        for k_local in 1:N_bi
            p = Int(param_idxs[k_local])
            (_, _, f_uid) = tjf_keys[p]
            f_col = dealer_to_col[f_uid]
            @inbounds for z in 1:Z
                M[k_local, z] = dist_zf[z, f_col]
            end
        end
        dist_blocks[bi] = M
    end

    # counts_blocks[bi]: SparseMatrixCSC{Int32, Int32}, N_bi × Z
    @info "Building counts_blocks"
    zip_to_col = Dict{String, Int}(z => i for (i, z) in enumerate(zips))
    I_pb = [Int32[] for _ in 1:n_blocks]
    J_pb = [Int32[] for _ in 1:n_blocks]
    V_pb = [Int32[] for _ in 1:n_blocks]
    for row in eachrow(counts)
        t_key = row.t::QuarterKey
        j_key = row.j::VehicleKey
        f_key = row.f::UUID
        p = get(product_lookup.tjf, (t_key, j_key, f_key), Int32(0))
        p == 0 && continue
        bi, k_local = tjf_to_block_k[p]
        col = get(zip_to_col, row.z, 0)
        col == 0 && continue
        push!(I_pb[bi], k_local)
        push!(J_pb[bi], Int32(col))
        push!(V_pb[bi], Int32(row.nrow))
    end
    counts_blocks = SparseMatrixCSC{Int32, Int32}[
        sparse(I_pb[bi], J_pb[bi], V_pb[bi], length(block_param_idxs[bi]), Z)
        for bi in 1:n_blocks
    ]

    # t_obs[bi, z] = column sums of counts_blocks[bi]
    t_obs = zeros(Int32, n_blocks, Z)
    @inbounds for bi in 1:n_blocks
        cb = counts_blocks[bi]
        for col in 1:Z
            for nz in cb.colptr[col]:(cb.colptr[col + 1] - 1)
                t_obs[bi, col] += cb.nzval[nz]
            end
        end
    end

    # D_dense (Dn × Z) and M_zs (Z) — in `zips` order
    D_dense = Matrix{Float32}(undef, Dn, Z)
    M_zs = Vector{Float32}(undef, Z)
    @inbounds for (zi, z) in enumerate(zips)
        ai = agent_data[z]
        for (di, attr) in enumerate(DEMOGRAPHIC_ATTRIBUTES)
            D_dense[di, zi] = Float32(getfield(ai, attr))
        end
        M_zs[zi] = Float32(ai.M)
    end

    inv_N = 1.0f0 / (sum(M_zs) * Float32(n_blocks))

    @info "load_model_frames done" N n_blocks Z K Dn

    return (;
        demo_scales,
        price_scale,
        dist_scale,
        distances,
        product_info,
        product_lookup,
        agent_data,
        zips,
        N,
        Z,
        K,
        Dn,
        n_blocks,
        block_keys,
        block_param_idxs,
        X_blocks,
        dist_blocks,
        counts_blocks,
        t_obs,
        D_dense,
        M_zs,
        tjf_to_block_k,
        inv_N,
    )
end
