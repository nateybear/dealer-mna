using DealerMnA

using Mmap
using StatsBase
using LinearAlgebra
using TensorOperations
using Memoize

using Optim
import ReverseDiff
import ADTypes
import LineSearches

include("bbl_matrix.jl")

# mc is hot; pre-allocate
@memoize mc_cached(T, dims...) = Array{T}(undef, dims)

function moment_scaling(w, f, t; C_j_map, C_f_map, C_t_map, R_map)
    mc = mc_cached(promote_type(eltype(w), eltype(f), eltype(t)), size(R_map)...)
    @tensor mc[i, j] = C_j_map[i, j, k] * w[k] + C_f_map[i, j, l] * f[l] + C_t_map[i, j, m] * t[m]
    W = R_map + mc
    return std(W, dims = 2)
end

function objective(w, f, t; C_j, C_t, C_f, R, scaling)
    mc = mc_cached(promote_type(eltype(w), eltype(f), eltype(t)), size(R, 1))
    mc .= C_j * w + C_f * f + C_t * t
    g = (R + mc) ./ scaling
    return norm(@. g * (g < 0))
end

function (@main)(_)
    (; index) = JLD2.load_object("data/bbl_matrix.jld2")

    n_row = length(index.tf_rows) * 150
    n_col = 100
    n_j = length(index.vehicle_cols)
    n_f = length(index.dealer_cols)
    n_t = 4

    C_j_map = Mmap.mmap("data/bbl_C_j.map", Array{Float64, 3}, (n_row, n_col, n_j))
    C_f_map = Mmap.mmap("data/bbl_C_f.map", Array{Float64, 3}, (n_row, n_col, n_f))
    C_t_map = Mmap.mmap("data/bbl_C_t.map", Array{Float64, 3}, (n_row, n_col, n_t))
    R_map = Mmap.mmap("data/bbl_R.map", Matrix{Float64}, (n_row, n_col))

    C_j = dropdims(mean(C_j_map, dims = 2), dims = 2)
    C_f = dropdims(mean(C_f_map, dims = 2), dims = 2)
    C_t = dropdims(mean(C_t_map, dims = 2), dims = 2)
    R = mean(R_map, dims = 2)

    scaling = zeros(size(R))

    w = ones(size(C_j, 2))
    f = zeros(size(C_f, 2))
    t = zeros(size(C_t, 2))

    function rescale!()
        scaling .= moment_scaling(w, f, t; C_j_map, C_f_map, C_t_map, R_map)
        return nothing
    end

    closed_objective = objective$(; C_j, C_t, C_f, R, scaling)

    w_objective = @$ closed_objective(_, f, t)
    f_objective = @$ closed_objective(w, _, t)
    t_objective = @$ closed_objective(w, f, _)

    function solve(f, init)
        return optimize(
            f,
            init,
            LBFGS(alphaguess = LineSearches.InitialStatic(scaled = true)),
            Optim.Options(show_trace = true),
            autodiff = ADTypes.AutoForwardDiff(chunksize = length(init)),
        ).minimizer
    end

    supnorm = maximum$abs

    i = 1
    tol = 1e-4
    while true
        @info "Round $i"

        rescale!()
        w_new = solve(w_objective, w)
        f_new = solve(f_objective, f)
        t_new = solve(t_objective, t)

        if supnorm(w - w_new) < tol && supnorm(f - f_new) < tol && supnorm(t - t_new) < tol
            @info "Convergence achieved; exiting"
            break
        end

        w .= w_new
        f .= f_new
        t .= t_new
    end

    JLD2.write_object("data/bbl_estimates.jld2", (; w, f, t))
    return nothing
end
