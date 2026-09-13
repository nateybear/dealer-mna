using DealerMnA
using .Demand
using DataFrames
using Statistics: mean, std

function (@main)(_)
    snakemake!()
    with_logging() do
        model = Demand.DemandModel(;
            JLD2.load_object(snakemake.input["model_frames"])...
        )
        price_table = read_arrow(snakemake.input["price_table"])

        # Mirrors bbl_forward_simulation_stratified.jl:96–104.
        vk_to_group = Dict{VehicleKey, Tuple{String, Int64, String}}()
        for row in eachrow(price_table)
            vk = (row.make, row.model, row.model_year, Float32(row.trim_rank))
            haskey(model.product_lookup.j, vk) || continue
            vk_to_group[vk] = (row.model, row.model_year, row.extra)
        end

        sim_paths = snakemake.input["simulations"]
        usable_vks = Set{VehicleKey}()
        sim_by_tf = Dict{Tuple{QuarterKey, UUID}, Vector{Vector{VehicleKey}}}()
        for path in sim_paths
            sim = JLD2.load_object(path)
            for ((t, f), nt) in sim
                push!(get!(sim_by_tf, (t, f), Vector{VehicleKey}[]), collect(nt.assortment))
                for vk in nt.assortment
                    push!(usable_vks, vk)
                end
            end
        end
        @info "Loaded sim assortments" n_seeds = length(sim_paths) n_usable_vks =
            length(usable_vks) n_tf = length(sim_by_tf)

        # Restrict empirical assortments to the sim's VK universe.
        intermed = Supply.tf_to_j_lookup(model)
        emp_by_tf = Dict{Tuple{QuarterKey, UUID}, Vector{VehicleKey}}()
        for ((t, f), vks) in intermed
            emp_by_tf[(t, f)] = [vk for vk in vks if vk ∈ usable_vks]
        end

        # Seed period's sim assortment equals the real one by construction
        # (bbl_forward_simulation_stratified.jl:259). Skip it — nothing to compare.
        all_ts = Set{QuarterKey}(t for ((t, _f), _) in sim_by_tf)
        t0 = minimum(all_ts)

        # -- 1. Assortment size per quarter -----------------------------------------
        by_q = Dict{QuarterKey, Vector{Tuple{Int, Vector{Int}}}}()
        n_skipped_no_emp = 0
        for ((t, f), sim_assorts) in sim_by_tf
            t == t0 && continue
            if !haskey(emp_by_tf, (t, f))
                n_skipped_no_emp += 1
                continue
            end
            n_emp = length(emp_by_tf[(t, f)])
            sim_sizes = length.(sim_assorts)
            push!(get!(by_q, t, Tuple{Int, Vector{Int}}[]), (n_emp, sim_sizes))
        end
        size_rows = DataFrame(
            year = Int[], quarter = Int[],
            n_emp_mean = Float64[], n_emp_std = Float64[],
            n_sim_mean = Float64[], n_sim_std = Float64[],
            frac_empty_emp = Float64[], frac_empty_sim = Float64[],
            n_dealer_quarters = Int[],
        )
        for (t, rows) in sort(collect(pairs(by_q)); by = first)
            emp_sizes = Int[r[1] for r in rows]
            sim_flat = reduce(vcat, Vector{Int}[r[2] for r in rows])
            push!(
                size_rows,
                (
                    t[1], t[2],
                    mean(emp_sizes), std(emp_sizes),
                    mean(sim_flat), std(sim_flat),
                    mean(emp_sizes .== 0), mean(sim_flat .== 0),
                    length(rows),
                ),
            )
        end
        @info "Assortment size (real vs sim, per quarter)" size_rows n_skipped_no_emp
        Arrow.write(snakemake.output["sizes"], size_rows)

        # -- 2. Per-group marginal carry rate ---------------------------------------
        groups = collect(Set(values(vk_to_group)))
        gi = Dict(g => i for (i, g) in enumerate(groups))
        G = length(groups)

        real_carries = zeros(Int, G)
        real_n_tf = 0
        for ((t, _f), vks) in emp_by_tf
            t == t0 && continue
            real_n_tf += 1
            gs = Set{Tuple{String, Int64, String}}()
            for vk in vks
                haskey(vk_to_group, vk) && push!(gs, vk_to_group[vk])
            end
            for g in gs
                real_carries[gi[g]] += 1
            end
        end
        sim_carries = zeros(Int, G)
        sim_n_tf = 0
        for ((t, _f), assorts) in sim_by_tf
            t == t0 && continue
            for assort in assorts
                sim_n_tf += 1
                gs = Set{Tuple{String, Int64, String}}()
                for vk in assort
                    haskey(vk_to_group, vk) && push!(gs, vk_to_group[vk])
                end
                for g in gs
                    sim_carries[gi[g]] += 1
                end
            end
        end
        group_table = DataFrame(
            model = String[], model_year = Int64[], extra = String[],
            p_real = Float64[], p_sim = Float64[], diff = Float64[],
        )
        for (g, i) in gi
            p_real = real_carries[i] / real_n_tf
            p_sim = sim_carries[i] / sim_n_tf
            push!(group_table, (g[1], g[2], g[3], p_real, p_sim, p_sim - p_real))
        end
        sort!(group_table, :diff)
        @info "Per-group carry-rate calibration" n_groups = G n_dq_real = real_n_tf n_dq_sim =
            sim_n_tf mean_abs_diff = mean(abs.(group_table.diff)) rms_diff = sqrt(
            mean(group_table.diff .^ 2),
        )
        Arrow.write(snakemake.output["groups"], group_table)

        # -- 3. Pairwise co-carry (independence check) -----------------------------
        # Excess over independent-Bernoulli baseline = P(A,B) − P(A)·P(B).
        # If sim's excess ≈ 0 but real's is substantially nonzero, marginals-only
        # (independent Bernoulli in the sim) is missing joint structure.
        real_pair = Dict{Tuple{Int, Int}, Int}()
        for ((t, _f), vks) in emp_by_tf
            t == t0 && continue
            g_idxs = sort!(
                unique!(Int[gi[vk_to_group[vk]] for vk in vks if haskey(vk_to_group, vk)]),
            )
            for i in 1:length(g_idxs), j in (i + 1):length(g_idxs)
                k = (g_idxs[i], g_idxs[j])
                real_pair[k] = get(real_pair, k, 0) + 1
            end
        end
        sim_pair = Dict{Tuple{Int, Int}, Int}()
        for ((t, _f), assorts) in sim_by_tf
            t == t0 && continue
            for assort in assorts
                g_idxs = sort!(
                    unique!(
                        Int[
                            gi[vk_to_group[vk]] for
                            vk in assort if haskey(vk_to_group, vk)
                        ],
                    ),
                )
                for i in 1:length(g_idxs), j in (i + 1):length(g_idxs)
                    k = (g_idxs[i], g_idxs[j])
                    sim_pair[k] = get(sim_pair, k, 0) + 1
                end
            end
        end
        pair_rows = DataFrame(
            model_A = String[], extra_A = String[], my_A = Int64[],
            model_B = String[], extra_B = String[], my_B = Int64[],
            n_real_AB = Int[],
            p_real_AB = Float64[], excess_real = Float64[],
            p_sim_AB = Float64[], excess_sim = Float64[],
        )
        # Only pairs with enough real co-carries to have signal.
        for ((a, b), n_r) in real_pair
            n_r < 20 && continue
            p_ra = real_carries[a] / real_n_tf
            p_rb = real_carries[b] / real_n_tf
            p_rab = n_r / real_n_tf
            p_sa = sim_carries[a] / sim_n_tf
            p_sb = sim_carries[b] / sim_n_tf
            p_sab = get(sim_pair, (a, b), 0) / sim_n_tf
            gA = groups[a]
            gB = groups[b]
            push!(
                pair_rows,
                (
                    gA[1], gA[3], gA[2],
                    gB[1], gB[3], gB[2],
                    n_r,
                    p_rab, p_rab - p_ra * p_rb,
                    p_sab, p_sab - p_sa * p_sb,
                ),
            )
        end
        sort!(pair_rows, :excess_real, rev = true)
        @info "Pairwise co-carry excess (positive = correlated beyond marginals)" n_pairs =
            nrow(pair_rows) mean_excess_real = mean(pair_rows.excess_real) mean_excess_sim = mean(
            pair_rows.excess_sim,
        ) mean_abs_gap = mean(abs.(pair_rows.excess_real .- pair_rows.excess_sim))
        Arrow.write(snakemake.output["pairs"], pair_rows)
    end
    return nothing
end
