# Standard errors for the non-δ demand parameters at the converged θ̂.
# Reuses the envelope-corrected concentrated Hessian H̃_NLL from src/demand/dense.jl,
# falls back to BHHH if the envelope computation throws. The asymptotic covariance is
# inv(H̃) and the standard errors are sqrt of its diagonal.

using DealerMnA
import .Demand:
    DemandModel, DemandWork,
    eval_combined!, eval_envelope_hessian!, definiteness_shift!, param_indices,
    build_nonδ_labels
using LinearAlgebra
using Distributions

snakemake!(stdin)
model = DemandModel(; JLD2.load_object(snakemake.input["model"])...)

with_logging() do
    θ = JLD2.load_object(snakemake.input["estimates"]).minimizer

    @info "Allocating DemandWork" Z = model.Z N_θ = length(θ)
    work = DemandWork(model)

    idx = param_indices(model)
    i_nonδ = collect(idx.i_Πᵦ[1]:idx.i_η[end])
    N_NONδ = length(i_nonδ)

    grad_buf = zeros(Float32, length(θ))
    bhhh_buf = zeros(Float32, N_NONδ, N_NONδ)
    @info "Evaluating combined pass at θ̂"
    eval_combined!(work, model, θ; grad = grad_buf, bhhh = bhhh_buf)

    H = zeros(Float32, N_NONδ, N_NONδ)
    used_bhhh = false
    try
        n_singular = eval_envelope_hessian!(work, model; H = H)
        @info "envelope Hessian computed" singular_blocks = n_singular
    catch e
        @warn "envelope Hessian failed ($e); falling back to BHHH"
        H .= bhhh_buf
        used_bhhh = true
    end
    # H and bhhh_buf are scaled by model.inv_N (Hessian of NLL_avg, not NLL_sum). Floor and
    # asymptotic variance both need to know this: floor scales with inv_N, and the MLE
    # asymptotic variance is inv(H_sum) = model.inv_N · inv(H_avg).
    shift = definiteness_shift!(H, 1.0f-3 * model.inv_N)
    @info "definiteness shift applied" shift = shift fallback_bhhh = used_bhhh

    V = Float64(model.inv_N) .* inv(Symmetric(Float64.(H)))
    diagV = diag(V)
    if any(<(0), diagV)
        @warn "negative diagonal entries in covariance" n_neg = count(<(0), diagV)
    end
    se = sqrt.(max.(diagV, 0.0))

    labels = build_nonδ_labels()
    df = DataFrame(
        name = labels,
        estimate = θ[i_nonδ],
        std_error = Float32.(se),
        p_val = 2 * cdf.(Normal(), -abs.(θ[i_nonδ] ./ se))
    )
    CSV.write(snakemake.output[1], df)
    @info "wrote standard errors" path = snakemake.output[1] nrow = nrow(df)
end
