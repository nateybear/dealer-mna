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

function (@main)(_)
    (; index) = JLD2.load_object("data/bbl_matrix.jld2")

    n_row = length(index.tf_rows) * 150
    n_col = 100
    n_j = length(index.vehicle_cols)
    n_f = length(index.dealer_cols)
    n_t = 4

    R_map = Mmap.mmap("data/bbl_R.map", Matrix{Float64}, (n_row, n_col))
    C_j_map = Mmap.mmap("data/bbl_C_j.map", Array{Float64, 3}, (n_row, n_col, n_j))
    C_f_map = Mmap.mmap("data/bbl_C_f.map", Array{Float64, 3}, (n_row, n_col, n_f))
    C_t_map = Mmap.mmap("data/bbl_C_t.map", Array{Float64, 3}, (n_row, n_col, n_t))

    R = dropdims(mean(R_map, dims = 2), dims = 2)
    C_j = dropdims(mean(C_j_map, dims = 2), dims = 2)
    C_f = dropdims(mean(C_f_map, dims = 2), dims = 2)
    C_t = dropdims(mean(C_t_map, dims = 2), dims = 2)

    # σ fixed at θ = 0 (mc contribution zero → std over sims of R alone).
    σ = vec(std(R_map, dims = 2))
    σ[σ .== 0] .= 1.0

    # SparseMatrixCSC is column-major; store the transpose so nzrange(_, i)
    # walks the nonzeros of C's row i as column i of the transpose.
    C_j_sp = sparse(C_j')
    C_f_sp = sparse(C_f')
    C_t_sp = sparse(C_t')

    # Recession-cone check for structurally-unbounded bound LPs. For the polytope
    # {θ : Cθ ≥ -R}, the recession cone is {d : Cd ≥ 0} (independent of R). If
    # column k of C is all ≤ 0, then d = -e_k satisfies Cd ≥ 0 → `min θ_k` is
    # unbounded below. Symmetric for max. Skip those LPs without touching HiGHS.
    function column_skip_flags(C)
        n = size(C, 2)
        skip_min = falses(n)
        skip_max = falses(n)
        for k in 1:n
            lo, hi = extrema(view(C, :, k))
            skip_min[k] = hi <= 0        # column ≤ 0 → -e_k in recession cone
            skip_max[k] =  lo >= 0        # column ≥ 0 → +e_k in recession cone
        end
        return skip_min, skip_max
    end
    w_skip_min, w_skip_max = column_skip_flags(C_j)
    f_skip_min, f_skip_max = column_skip_flags(C_f)
    t_skip_min, t_skip_max = column_skip_flags(C_t)

    model = Model(HiGHS.Optimizer)
    set_silent(model)
    set_attribute(model, "parallel", "on")
    set_attribute(model, "threads", nthreads())
    # Half the bound LPs hit DUAL_INFEASIBLE by design (one-sided BBL moments);
    # HiGHS's default ray-extraction on those is wasted work.
    set_attribute(model, HiGHS.ComputeInfeasibilityCertificate(), false)

    @variable(model, w[1:n_j])
    @variable(model, f[1:n_f])
    @variable(model, t[1:n_t])

    # mc = w[j] + f[dealer] + t[quarter] has a 2D additive null space:
    # (w + c₁, f + c₂, t + c₃) with c₁+c₂+c₃ = 0 leaves every moment invariant.
    # Pin two coordinates to eliminate; every other coordinate is then a
    # relative effect vs these reference categories.
    @constraint(model, f[1] == 0)
    @constraint(model, t[1] == 0)

    for i in 1:n_row
        s = 1 / σ[i]
        expr = AffExpr(R[i] * s)
        add_sparse_row!(expr, C_j_sp, i, w, s)
        add_sparse_row!(expr, C_f_sp, i, f, s)
        add_sparse_row!(expr, C_t_sp, i, t, s)
        @constraint(model, expr >= 0)
    end

    @info "Solving feasibility LP" n_row n_j n_f n_t
    set_objective_sense(model, FEASIBILITY_SENSE)
    optimize!(model)
    ts = termination_status(model)
    ts in (OPTIMAL, LOCALLY_SOLVED) || error("BBL polytope infeasible: $ts")
    @info "Feasibility OK; solving bounding-box LPs" n_bounds = 2 * (n_j + n_f + n_t)

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

    function bounds_for(name, xs, n, skip_min, skip_max)
        out = Vector{Tuple{Float64, Float64}}(undef, n)
        finite_lo = 0
        finite_hi = 0
        n_skipped_min = 0
        n_skipped_max = 0
        for k in 1:n
            lo = if skip_min[k]
                n_skipped_min += 1
                -Inf
            else
                solve_bound!(MIN_SENSE, xs[k])
            end
            hi = if skip_max[k]
                n_skipped_max += 1
                Inf
            else
                solve_bound!(MAX_SENSE, xs[k])
            end
            out[k] = (lo, hi)
            isfinite(lo) && (finite_lo += 1)
            isfinite(hi) && (finite_hi += 1)
            @info "$name[$k]" lo hi
        end
        @info "$name bounds done" n finite_lo finite_hi n_skipped_min n_skipped_max
        return out
    end

    w_bounds = bounds_for("w", w, n_j, w_skip_min, w_skip_max)
    f_bounds = bounds_for("f", f, n_f, f_skip_min, f_skip_max)
    t_bounds = bounds_for("t", t, n_t, t_skip_min, t_skip_max)

    JLD2.write_object("data/bbl_estimates.jld2", (; w = w_bounds, f = f_bounds, t = t_bounds))
    return nothing
end
