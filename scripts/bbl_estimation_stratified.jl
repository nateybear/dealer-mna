using Distributed
@everywhere using JuMP #load macros first b/c of eval order
@everywhere using DealerMnA

# used in main
macro fetchinit(args...)
    exprs = [:($(esc(a)) = @fetch $(esc(a))) for a in args]
    quote
        $(exprs...)
    end
end

@everywhere workers() begin
    using Mmap
    using SparseArrays
    using Statistics
    using Base.Threads

    using JuMP
    using HiGHS
    using KNITRO

    function add_sparse_row!(expr::AffExpr, C_sp::SparseMatrixCSC, i::Int, x, scale)
        rows = rowvals(C_sp)
        vals = nonzeros(C_sp)
        for k in nzrange(C_sp, i)
            add_to_expression!(expr, vals[k] * scale, x[rows[k]])
        end
        return nothing
    end

    # Accumulate a per-sim source slice into `dest`, permuting scenarios within
    # each tf according to `perms[tf_idx]`. Scenario-outer layout: rank-k row
    # for tf_idx lives at `tf_idx + (k-1)*n_tf`; its raw source is at
    # `tf_idx + (perms[tf_idx][k]-1)*n_tf`.
    function accumulate_ranked!(dest, src, perms, n_tf, n_scen)
        for tf_idx in 1:n_tf, k in 1:n_scen
            canonical_row = tf_idx + (k - 1) * n_tf
            raw_row = tf_idx + (perms[tf_idx][k] - 1) * n_tf
            dest[canonical_row] += src[raw_row]
        end
        return nothing
    end

    (; index) = JLD2.load_object("data/bbl_matrix_stratified.jld2")

    # Map layout per sim column: SIM_SIZE per-product local perturbations first,
    # then N_SHIFTS coordinated uniform shifts. Both blocks share the same
    # (rank, tf_idx) row indexing pattern used by accumulate_ranked!.
    const N_LOCAL = Supply.SIM_SIZE
    const N_SHIFTS = length(Supply.COORDINATED_SHIFTS)
    const N_TOTAL = N_LOCAL + N_SHIFTS

    n_tf = length(index.tf_rows)
    n_col = 20                                 # BBL sim replications; matches Snakefile expand range
    n_j = length(index.vehicle_cols)
    n_f = length(index.dealer_cols)
    n_t = 4
    n_row = n_tf * N_TOTAL

    R_map = Mmap.mmap("data/bbl_R_stratified.map", Matrix{Float64}, (n_row, n_col))
    C_j_map = Mmap.mmap("data/bbl_C_j_stratified.map", Array{Float64, 3}, (n_row, n_col, n_j))
    C_f_map = Mmap.mmap("data/bbl_C_f_stratified.map", Array{Float64, 3}, (n_row, n_col, n_f))
    C_t_map = Mmap.mmap("data/bbl_C_t_stratified.map", Array{Float64, 3}, (n_row, n_col, n_t))

    # Block spec: (name, n_scen, map_row_lo, map_row_hi, sorted)
    # - local×sorted:   attacks noise cancellation (rank-k averaged across sims has consistent E[·] ≠ 0)
    # - local×raw:      cheap extra constraint; disciplines sampling error
    # - coord×sorted:   ranks the coordinated shifts by revenue diff before averaging
    # - coord×raw:      shift α=0.5 always in slot 1 etc.; interpretable per-scenario means
    local_map_lo = 1
    local_map_hi = N_LOCAL * n_tf
    coord_map_lo = local_map_hi + 1
    coord_map_hi = n_row

    block_specs = (
        (:local_sorted, N_LOCAL, local_map_lo, local_map_hi, true),
        (:local_raw, N_LOCAL, local_map_lo, local_map_hi, false),
        (:coord_sorted, N_SHIFTS, coord_map_lo, coord_map_hi, true),
        (:coord_raw, N_SHIFTS, coord_map_lo, coord_map_hi, false),
    )

    R_blocks = [zeros(Float64, n_tf * spec[2]) for spec in block_specs]
    Cj_blocks = [zeros(Float64, n_tf * spec[2], n_j) for spec in block_specs]
    Cf_blocks = [zeros(Float64, n_tf * spec[2], n_f) for spec in block_specs]
    Ct_blocks = [zeros(Float64, n_tf * spec[2], n_t) for spec in block_specs]

    @info "Accumulating 4 moment blocks" n_col n_tf N_LOCAL N_SHIFTS
    for s in 1:n_col
        R_slice = R_map[:, s]
        for (bi, (name, n_scen, lo, hi, sorted)) in enumerate(block_specs)
            R_block = @view R_slice[lo:hi]
            perms = if sorted
                [sortperm(@view R_block[tf_idx .+ (0:(n_scen - 1)) .* n_tf]) for tf_idx in 1:n_tf]
            else
                [collect(1:n_scen) for _ in 1:n_tf]
            end
            accumulate_ranked!(R_blocks[bi], R_block, perms, n_tf, n_scen)
            for j in 1:n_j
                accumulate_ranked!(
                    @view(Cj_blocks[bi][:, j]), @view(C_j_map[lo:hi, s, j]), perms, n_tf, n_scen
                )
            end
            for j in 1:n_f
                accumulate_ranked!(
                    @view(Cf_blocks[bi][:, j]), @view(C_f_map[lo:hi, s, j]), perms, n_tf, n_scen
                )
            end
            for j in 1:n_t
                accumulate_ranked!(
                    @view(Ct_blocks[bi][:, j]), @view(C_t_map[lo:hi, s, j]), perms, n_tf, n_scen
                )
            end
        end
    end
    for bi in eachindex(block_specs)
        R_blocks[bi] ./= n_col
        Cj_blocks[bi] ./= n_col
        Cf_blocks[bi] ./= n_col
        Ct_blocks[bi] ./= n_col
    end

    # SparseMatrixCSC is column-major; store the transpose so nzrange(_, i)
    # walks the nonzeros of C's row i as column i of the transpose.
    Cj_sps = [sparse(C') for C in Cj_blocks]
    Cf_sps = [sparse(C') for C in Cf_blocks]
    Ct_sps = [sparse(C') for C in Ct_blocks]

    # Recession-cone check across the stacked constraint set: for the polytope
    # {θ : Cθ ≥ -R}, `min θ_k` is unbounded below iff column k is ≤ 0 across
    # EVERY block; otherwise some row hits binding. Take extrema jointly.
    function column_skip_flags(C_list)
        n = size(C_list[1], 2)
        skip_min = falses(n)
        skip_max = falses(n)
        for k in 1:n
            lo, hi = extrema(view(C_list[1], :, k))
            for i in 2:length(C_list)
                lo2, hi2 = extrema(view(C_list[i], :, k))
                lo = min(lo, lo2)
                hi = max(hi, hi2)
            end
            skip_min[k] = hi <= 0
            skip_max[k] = lo >= 0
        end
        return skip_min, skip_max
    end
    w_skip_min, w_skip_max = column_skip_flags(Cj_blocks)
    f_skip_min, f_skip_max = column_skip_flags(Cf_blocks)
    t_skip_min, t_skip_max = column_skip_flags(Ct_blocks)

    model = Model(KNITRO.Optimizer)
    set_silent(model)
    #set_attribute(model, "threads", nthreads())
    # Half the bound LPs hit DUAL_INFEASIBLE by design (one-sided BBL moments);
    # HiGHS's default ray-extraction on those is wasted work.
    #set_attribute(model, HiGHS.ComputeInfeasibilityCertificate(), false)

    @variable(model, 0 <= w[1:n_j] <= 1000)
    @variable(model, -1000 <= f[1:n_f] <= 1000)
    @variable(model, -1000 <= t[1:n_t] <= 1000)
    # treat f and t as nuisance FEs, sum-to-zero constraint
    # identifies the levels of wholesale price
    @constraint(model, sum(f) == 0)
    @constraint(model, sum(t) == 0)

    n_moments = sum(length(R_blocks[bi]) for bi in eachindex(block_specs))
    @variable(model, ϵ[1:n_moments] ≥ 0)

    total_constraints = sum(length(R) for R in R_blocks)
    @info "Adding LP constraints" total_constraints per_block = [length(R) for R in R_blocks]
    ϵ_rest = ϵ
    for bi in eachindex(block_specs)
        R_b = R_blocks[bi]
        Cj_sp = Cj_sps[bi]
        Cf_sp = Cf_sps[bi]
        Ct_sp = Ct_sps[bi]
        for i in 1:length(R_b)
            global ϵ_rest
            ϵ_i, ϵ_rest = Iterators.peel(ϵ_rest)
            s = 1.0
            expr = AffExpr(R_b[i] * s)
            add_sparse_row!(expr, Cj_sp, i, w, s)
            add_sparse_row!(expr, Cf_sp, i, f, s)
            add_sparse_row!(expr, Ct_sp, i, t, s)
            @constraint(model, expr + ϵ_i ≥ 0)
        end
    end

    # Flat metadata aligned to ε ordering (block1 rows, block2 rows, ...).
    # Row i inside block bi's stripe corresponds to (tf_idx, k) via
    #   tf_idx = ((i-1) mod n_tf) + 1, k = ((i-1) ÷ n_tf) + 1.
    # k = rank position for sorted blocks, raw scenario index for raw blocks.
    row_meta = let meta = Vector{NamedTuple{(:block, :t, :k), Tuple{Symbol, Demand.QuarterKey, Int}}}()
        for (bi, (name, n_scen, _lo, _hi, _sorted)) in enumerate(block_specs)
            for i in 1:(n_tf * n_scen)
                tf_idx = ((i - 1) % n_tf) + 1
                k = ((i - 1) ÷ n_tf) + 1
                tf = index.tf_rows[tf_idx]
                push!(meta, (block = name, t = tf[1], k = k))
            end
        end
        meta
    end

    function solve_bound!(sense, obj)
        set_objective_sense(model, sense)
        set_objective_function(model, obj)
        model_copy = copy(model)
        JuMP.set_optimizer(model_copy, KNITRO.Optimizer)
        set_attribute(model_copy, "linsolver", 7)
	set_silent(model_copy)
        optimize!(model_copy)
        assert_is_solved_and_feasible(model_copy)
        return objective_value(model_copy), value.(all_variables(model_copy))
    end

    add_sse_constraint!(c) = @constraint(model, sum(ϵ .^ 2) ≤ c)

    const SIGMA_FLOOR = 1e-8
    weighted_sse_obj(σ) = sum((1 / max(σ[i], SIGMA_FLOOR)^2) * ϵ[i]^2 for i in 1:length(ϵ))
    add_weighted_sse_constraint!(c, σ) =
        @constraint(model, sum((1 / max(σ[i], SIGMA_FLOOR)^2) * ϵ[i]^2 for i in 1:length(ϵ)) ≤ c)

    set_start_values!(v) = set_start_value.(all_variables(model), v)

    # Per-moment σ under (w, f, t) start values. Reproduces the per-block
    # accumulate_ranked pattern but tracks per-sim values instead of averaging,
    # so we can compute std over sims of the constraint LHS at start_vals.
    function compute_moment_sigmas(w_start, f_start, t_start)
        flat = Float64[]
        for (bi, (name, n_scen, lo, hi, sorted)) in enumerate(block_specs)
            n_rows = n_tf * n_scen
            s_sum = zeros(Float64, n_rows)
            s_sq = zeros(Float64, n_rows)
            for s in 1:n_col
                R_slice = R_map[lo:hi, s]
                Cj_slice = C_j_map[lo:hi, s, :]
                Cf_slice = C_f_map[lo:hi, s, :]
                Ct_slice = C_t_map[lo:hi, s, :]
                vals_raw = R_slice .+ Cj_slice * w_start .+
                           Cf_slice * f_start .+ Ct_slice * t_start
                perms = if sorted
                    [
                        sortperm(@view R_slice[tf_idx .+ (0:(n_scen - 1)) .* n_tf])
                        for tf_idx in 1:n_tf
                    ]
                else
                    [collect(1:n_scen) for _ in 1:n_tf]
                end
                for tf_idx in 1:n_tf, k in 1:n_scen
                    canonical_row = tf_idx + (k - 1) * n_tf
                    raw_row = tf_idx + (perms[tf_idx][k] - 1) * n_tf
                    v = vals_raw[raw_row]
                    s_sum[canonical_row] += v
                    s_sq[canonical_row] += v * v
                end
            end
            means = s_sum ./ n_col
            vars = max.(0.0, (s_sq ./ n_col) .- means .^ 2)
            append!(flat, sqrt.(vars))
        end
        return flat
    end

    # Wraps solve_bound! and additionally returns the ε values from the copy,
    # used by the constraint-attribution diagnostic to identify which rows pin
    # each solved bound.
    function solve_bound_with_epsilons!(sense, obj)
        set_objective_sense(model, sense)
        set_objective_function(model, obj)
        model_copy = copy(model)
        JuMP.set_optimizer(model_copy, KNITRO.Optimizer)
        set_attribute(model_copy, "linsolver", 7)
        set_silent(model_copy)
        optimize!(model_copy)
        assert_is_solved_and_feasible(model_copy)
        all_vals = value.(all_variables(model_copy))
        # Variable ordering: w (n_j), f (n_f), t (n_t), ε (n_moments).
        ε_vals = all_vals[(n_j + n_f + n_t + 1):end]
        return objective_value(model_copy), all_vals, ε_vals
    end
end

@fetchinit n_j w_skip_min w_skip_max n_f f_skip_min f_skip_max n_t t_skip_min t_skip_max total_constraints

@everywhere workers() begin
    # Attribute a solved bound to its binding constraint rows. Returns
    # ((block, quarter) => count) sorted by count descending, plus total
    # binding count and max_ε for context.
    function attribute_binding(ε_vals::AbstractVector)
        max_ε = maximum(ε_vals)
        if max_ε < 1e-12
            return Tuple{Tuple{Symbol, Demand.QuarterKey}, Int}[], 0, max_ε
        end
        threshold = 0.9 * max_ε
        counts = Dict{Tuple{Symbol, Demand.QuarterKey}, Int}()
        n_binding = 0
        for i in eachindex(ε_vals)
            ε_vals[i] >= threshold || continue
            n_binding += 1
            m = row_meta[i]
            key = (m.block, m.t)
            counts[key] = get(counts, key, 0) + 1
        end
        ranked = sort(collect(counts); by = last, rev = true)
        return ranked, n_binding, max_ε
    end

    format_binding_summary(bindings) = isempty(bindings) ? "-" :
        join(
            (
                "$(k[1]) $(k[2][1])Q$(k[2][2]):$(v)"
                for (k, v) in Iterators.take(bindings, 3)
            ),
            ", ",
        )
end

function (@main)(_)
    @info "Step 1: unweighted feasibility" total_constraints n_j n_f n_t
    c_unweighted, start_vals = @fetch solve_bound!(MIN_SENSE, sum(ϵ .^ 2))
    @info "Unweighted feasibility done" c_unweighted

    w_start = start_vals[1:n_j]
    f_start = start_vals[(n_j + 1):(n_j + n_f)]
    t_start = start_vals[(n_j + n_f + 1):(n_j + n_f + n_t)]

    @info "Step 2: per-moment σ at unweighted feasibility solution"
    σ = @fetch compute_moment_sigmas(w_start, f_start, t_start)
    let σ_sorted = sort(σ)
        n = length(σ_sorted)
        @info "σ summary" σ_min = σ_sorted[1] σ_q10 = σ_sorted[max(1, n ÷ 10)] σ_median = σ_sorted[max(1, n ÷ 2)] σ_q90 = σ_sorted[max(1, 9 * n ÷ 10)] σ_max = σ_sorted[end] n_below_floor = count(<(1e-8), σ)
    end

    @info "Step 3: weighted feasibility"
    c_weighted, start_vals_weighted = @fetch solve_bound!(MIN_SENSE, weighted_sse_obj($σ))
    @info "Weighted feasibility done" c_weighted c_unweighted ratio_c = c_weighted / max(c_unweighted, eps())

    @everywhere workers() set_start_values!($start_vals_weighted)
    @everywhere workers() add_weighted_sse_constraint!($c_weighted, $σ)

    @info "Step 4: bounding-box LPs on weighted problem" n_bounds = 2 * (n_j + n_f + n_t)

    function bounds_for(name, xs, n, skip_min, skip_max)
        pmap(1:n) do k
            x = model[xs][k]
            lo, lo_bindings, lo_n = if skip_min[k]
                (-Inf, Tuple{Tuple{Symbol, Demand.QuarterKey}, Int}[], 0)
            else
                v, _, εs = solve_bound_with_epsilons!(MIN_SENSE, x)
                bindings, n_binding, _max_ε = attribute_binding(εs)
                (v, bindings, n_binding)
            end
            hi, hi_bindings, hi_n = if skip_max[k]
                (Inf, Tuple{Tuple{Symbol, Demand.QuarterKey}, Int}[], 0)
            else
                v, _, εs = solve_bound_with_epsilons!(MAX_SENSE, x)
                bindings, n_binding, _max_ε = attribute_binding(εs)
                (v, bindings, n_binding)
            end
            extra = if xs == :w
                " $(index.vehicle_cols[k])"
            else
                ""
            end
            @info "$name[$k]$extra" lo hi n_bind_lo = lo_n n_bind_hi = hi_n top_lo = format_binding_summary(
                lo_bindings,
            ) top_hi = format_binding_summary(hi_bindings)
            return (; lo, hi, lo_bindings, hi_bindings)
        end
    end

    w_bounds = bounds_for("w", :w, n_j, w_skip_min, w_skip_max)
    f_bounds = bounds_for("f", :f, n_f, f_skip_min, f_skip_max)
    t_bounds = bounds_for("t", :t, n_t, t_skip_min, t_skip_max)

    JLD2.write_object(
        "data/bbl_estimates_stratified.jld2",
        (;
            w = w_bounds, f = f_bounds, t = t_bounds,
            c_unweighted, c_weighted, σ,
        ),
    )
    return nothing
end
