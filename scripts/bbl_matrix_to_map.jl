using DealerMnA
using Mmap
using SparseArrays
import SparseArrays: findnz
using Base.Threads

# Convert the BBLMatrices JLD2 (nested V/M cells indexed by (scenario, sim)) into
# flat memory-mapped Float64 arrays that scripts/bbl_estimation.jl consumes.
#
# For each per-simulation column:
#   - Row 1 of BBLMatrices is the observed policy; rows 2..end are perturbations.
#   - Output stores observed − perturbation (the BBL moment difference).
# Row layout in each .map file: `(iscen - 1) * n_tf + tf_idx`. `iscen` cycles
# over the perturbation index (1..n_scen). n_scen = outerM − 1 =
# Supply.SIM_SIZE + N_SHIFTS from bbl_matrix.jl.

function write_R!(io, mat, n_tf, n_sims, n_scen)
    A = Mmap.mmap(io, Matrix{Float64}, (n_tf * n_scen, n_sims))
    @inbounds for iscen in 1:n_scen
        m_start = (iscen - 1) * n_tf
        @threads for s in 1:n_sims
            obs = mat[1, s]
            pert = mat[iscen + 1, s]
            for tf_idx in 1:n_tf
                A[m_start + tf_idx, s] = Float64(obs[tf_idx] - pert[tf_idx])
            end
        end
    end
    Mmap.sync!(A)
    return nothing
end

function write_C!(io, mat, n_tf, n_sims, n_scen, n_fe)
    A = Mmap.mmap(io, Array{Float64, 3}, (n_tf * n_scen, n_sims, n_fe))
    @inbounds for iscen in 1:n_scen
        m_start = (iscen - 1) * n_tf
        @threads for s in 1:n_sims
            sm = mat[1, s] - mat[iscen + 1, s]   # SparseMatrixCSC subtraction
            rows, cols, vals = findnz(sm)
            for k in eachindex(rows)
                A[m_start + rows[k], s, cols[k]] = vals[k]
            end
        end
    end
    Mmap.sync!(A)
    return nothing
end

function (@main)(_)
    snakemake!()
    with_logging() do
        mat = JLD2.load_object(snakemake.input["bbl_matrix"])
        idx = mat.index
        n_tf = length(idx.tf_rows)
        n_sims = size(mat.R_f, 2)
        n_scen = size(mat.R_f, 1) - 1
        n_j = length(idx.vehicle_cols)
        n_f = length(idx.dealer_cols)
        n_t = 4
        @info "Converting BBLMatrices → mmap diffs" n_tf n_sims n_scen n_j n_f n_t

        # `open(_, "w+")` truncates and gives read+write; `Mmap.mmap` on the
        # returned IO then extends the file to the mapped size.
        open(snakemake.output["R"], "w+") do io
            write_R!(io, mat.R_f, n_tf, n_sims, n_scen)
        end
        open(snakemake.output["C_j"], "w+") do io
            write_C!(io, mat.C_j, n_tf, n_sims, n_scen, n_j)
        end
        open(snakemake.output["C_f"], "w+") do io
            write_C!(io, mat.C_f, n_tf, n_sims, n_scen, n_f)
        end
        open(snakemake.output["C_t"], "w+") do io
            write_C!(io, mat.C_t, n_tf, n_sims, n_scen, n_t)
        end
    end
    return nothing
end
