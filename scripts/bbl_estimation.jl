using Distributed
@everywhere using JuMP #load macros first b/c of eval order

@everywhere begin
    using DealerMnA

    using Mmap
    using SparseArrays
    using Statistics
    using Base.Threads

    using JuMP
    using HiGHS

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

    (; index) = JLD2.load_object("data/bbl_matrix.jld2")

    # Map layout per sim column: SIM_SIZE per-product local perturbations first,
    # then N_SHIFTS coordinated uniform shifts. Both blocks share the same
    # (rank, tf_idx) row indexing pattern used by accumulate_ranked!.
    const N_LOCAL = Supply.SIM_SIZE
    const N_SHIFTS = 8                          # must match COORDINATED_SHIFTS in bbl_forward_simulation.jl
    const N_TOTAL = N_LOCAL + N_SHIFTS

    n_tf = length(index.tf_rows)
    n_col = 200                                 # BBL sim replications; matches Snakefile expand range
    n_j = length(index.vehicle_cols)
    n_f = length(index.dealer_cols)
    n_t = 4
    n_row = n_tf * N_TOTAL

    R_map = Mmap.mmap("data/bbl_R.map", Matrix{Float64}, (n_row, n_col))
    C_j_map = Mmap.mmap("data/bbl_C_j.map", Array{Float64, 3}, (n_row, n_col, n_j))
    C_f_map = Mmap.mmap("data/bbl_C_f.map", Array{Float64, 3}, (n_row, n_col, n_f))
    C_t_map = Mmap.mmap("data/bbl_C_t.map", Array{Float64, 3}, (n_row, n_col, n_t))

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

    # Per-map-row std across sims. Sorted blocks reuse the raw-row σ at the
    # matching canonical position: it's a scale conditioner, not a statistical
    # estimand of the accumulated cell, and using the same σ for both orderings
    # keeps the LP conditioning consistent across the block pair.
    σ_map = vec(std(R_map, dims = 2))
    σ_map[σ_map .== 0] .= 1.0

    σ_blocks = [σ_map[spec[3]:spec[4]] for spec in block_specs]

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

    model = Model(HiGHS.Optimizer)
    set_silent(model)
    set_attribute(model, "parallel", "on")
    set_attribute(model, "threads", nthreads())
    # Half the bound LPs hit DUAL_INFEASIBLE by design (one-sided BBL moments);
    # HiGHS's default ray-extraction on those is wasted work.
    set_attribute(model, HiGHS.ComputeInfeasibilityCertificate(), false)

    @variable(model, 0 <= w[1:n_j] <= 1000)
    @variable(model, -1000 <= f[1:n_f] <= 1000)
    @variable(model, -1000 <= t[1:n_t] <= 1000)
    n_moments = sum(length(R_blocks[bi]) for bi in eachindex(block_specs))
    @variable(model, ϵ[1:n_moments] ≥ 0)

    # mc = w[j] + f[dealer] + t[quarter] has a 2D additive null space:
    # (w + c₁, f + c₂, t + c₃) with c₁+c₂+c₃ = 0 leaves every moment invariant.
    # Pin two coordinates to eliminate; every other coordinate is then a
    # relative effect vs these reference categories.
    @constraint(model, f[1] == 0)
    @constraint(model, t[1] == 0)

    total_constraints = sum(length(R) for R in R_blocks)
    @info "Adding LP constraints" total_constraints per_block = [length(R) for R in R_blocks]
    ϵ_rest = ϵ
    for bi in eachindex(block_specs)
        R_b = R_blocks[bi]
        σ_b = σ_blocks[bi]
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

    function solve_bound!(sense, obj)
        set_objective_sense(model, sense)
        set_objective_function(model, obj)
        optimize!(model)
        s = termination_status(model)
        s == OPTIMAL && return objective_value(model)
        # BBL moment inequalities are one-sided so unbounded directions are
        # expected: sending mc → -∞ trivially satisfies `R + Cθ ≥ 0`.
        s == DUAL_INFEASIBLE && return sense == MIN_SENSE ? -Inf : Inf
        return error("unexpected LP status: $s")
    end

    add_sse_constraint!(c) = @constraint(model, sum(ϵ .^ 2) ≤ c)
end

function (@main)(_)
    @info "Solving Initial Bound" total_constraints n_j n_f n_t
    c = solve_bound!(MIN_SENSE, sum(ϵ .^ 2))
    ts = termination_status(model)
    ts in (OPTIMAL, LOCALLY_SOLVED) || error("BBL polytope infeasible: $ts")
    @info "Feasibility OK; solving bounding-box LPs" n_bounds = 2 * (n_j + n_f + n_t) c

    @everywhere add_sse_constraint!(c)

    bounds_for(name, xs, n, skip_min, skip_max) =
        pmap(1:n) do k
            lo = if skip_min[k]
                -Inf
            else
                solve_bound!(MIN_SENSE, xs[k])
            end
            hi = if skip_max[k]
                Inf
            else
                solve_bound!(MAX_SENSE, xs[k])
            end
            @info "$name[$k]" lo hi
            return (lo, hi)
        end

    w_bounds = bounds_for("w", w, n_j, w_skip_min, w_skip_max)
    f_bounds = bounds_for("f", f, n_f, f_skip_min, f_skip_max)
    t_bounds = bounds_for("t", t, n_t, t_skip_min, t_skip_max)

    JLD2.write_object("data/bbl_estimates.jld2", (; w = w_bounds, f = f_bounds, t = t_bounds))
    return nothing
end
