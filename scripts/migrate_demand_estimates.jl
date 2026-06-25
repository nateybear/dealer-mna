# One-shot migration: re-order the δ block of data/demand_estimates.jld2 to match the
# new DemandModel ordering (sorted by (t, j, f)) introduced by the 2026-06 Demand
# refactor. Run AFTER rebuilding data/model_frames.jld2 with the new code:
#
#   snakemake -F data/model_frames.jld2
#   julia --project=. scripts/migrate_demand_estimates.jl
#
# The old δ ordering is recovered from data/delta_decomposition_regout.jld2 — its `df`
# was constructed by iterating `pairs(model.product_lookup.tjf)` in OLD insertion
# order, so row `i` of df carries the (t, j, f) triple that lived at old δ position `i`.
# The 22 Πᵦ/Πp/η parameters are at the same positions in old and new (ORDER unchanged),
# so they pass through unmodified.
#
# Side effects:
#   - Writes backup data/demand_estimates.old_ordering.jld2 (skipped if it already exists)
#   - Overwrites data/demand_estimates.jld2 with the reordered fit
#
# This script is NOT part of the Snakefile DAG — it's a one-off helper.

using DealerMnA
using DataFrames

const OLD_ESTIMATES = "data/demand_estimates.jld2"
const DECOMP = "data/delta_decomposition_regout.jld2"
const NEW_MODEL = "data/model_frames.jld2"
const BACKUP = "data/demand_estimates.old_ordering.jld2"

isfile(OLD_ESTIMATES) || error("missing $OLD_ESTIMATES")
isfile(DECOMP) || error("missing $DECOMP — required to recover the OLD δ ordering")
isfile(NEW_MODEL) || error("missing $NEW_MODEL — rebuild first: snakemake -F $NEW_MODEL")

if isfile(BACKUP)
    @info "backup already exists, leaving it in place" backup = BACKUP
else
    @info "backing up old fit" from = OLD_ESTIMATES to = BACKUP
    cp(OLD_ESTIMATES, BACKUP)
end

# 1. Load OLD ordering from delta_decomposition df. Row i ↔ old δ position i.
@info "loading old δ ordering from $DECOMP"
decomp = JLD2.load_object(DECOMP)
df = decomp.df
N_old = nrow(df)
old_key_to_idx = Dict{Demand.TJFKey, Int}()
for (i, row) in enumerate(eachrow(df))
    t = (Int64(row.year), Int64(row.quarter))
    j = (String(row.make), String(row.model), Int64(row.model_year), Float32(row.trim_rank))
    f = row.dealer_uid
    old_key_to_idx[(t, j, f)] = i
end
@info "old δ count" N_old length(old_key_to_idx)

# 2. Load NEW model, extract new (t, j, f) → new δ position mapping.
@info "loading new model from $NEW_MODEL"
new_nt = JLD2.load_object(NEW_MODEL)
new_model = try
    Demand.DemandModel(; new_nt...)
catch e
    error(
        "Failed to build DemandModel from $NEW_MODEL — is it in the new format? " *
        "Rebuild with `snakemake -F $NEW_MODEL`. (inner error: $e)",
    )
end
new_tjf = new_model.product_lookup.tjf
N_new = length(new_tjf)
@info "new δ count" N_new

# 3. Validate key-set overlap (warn-only — handle gracefully below).
new_keys = Set(keys(new_tjf))
old_keys = Set(keys(old_key_to_idx))
only_in_old = setdiff(old_keys, new_keys)
only_in_new = setdiff(new_keys, old_keys)
if !isempty(only_in_old)
    @warn "old fit had (t,j,f) triples no longer in the model" n_dropped = length(only_in_old)
end
if !isempty(only_in_new)
    @warn "new model has (t,j,f) triples missing from old fit; their δ will start at -10.0" n_added = length(
        only_in_new
    )
end

# 4. Build permutation: perm[new_idx] = old_idx, or 0 if no old match.
perm = zeros(Int, N_new)
for (key, new_idx) in pairs(new_tjf)
    perm[Int(new_idx)] = get(old_key_to_idx, key, 0)
end

# 5. Load old fit, split into [δ; π]. Verify lengths.
@info "loading old fit from $OLD_ESTIMATES"
fit = JLD2.load_object(OLD_ESTIMATES)
old_minimizer = Float32.(getproperty(fit, :minimizer))
length(old_minimizer) == N_old + 22 ||
    error("old minimizer length $(length(old_minimizer)) ≠ N_old + 22 = $(N_old + 22)")
old_δ = old_minimizer[1:N_old]
old_π = old_minimizer[(N_old + 1):end]

# Sanity: same Πᵦ/Πp/η count in new model.
idx = Demand.param_indices(new_model)
n_nonδ_new = length(idx.i_Πᵦ) + length(idx.i_Πp) + length(idx.i_η)
n_nonδ_new == 22 ||
    error("new model has $n_nonδ_new non-δ params, expected 22 — π block migration is unsafe")

# 6. Reorder δ.
new_δ = fill(-10.0f0, N_new)  # initial_parameters default for unmapped products
n_filled = 0
for new_idx in 1:N_new
    old_idx = perm[new_idx]
    old_idx == 0 && continue
    new_δ[new_idx] = old_δ[old_idx]
    global n_filled += 1
end
@info "reordered δ" n_filled n_init_default = N_new - n_filled

# 7. Save new fit. Wrap as a NamedTuple compatible with the estimation script's
# `result.minimizer` access path.
new_minimizer = vcat(new_δ, old_π)
@info "writing reordered fit" path = OLD_ESTIMATES N_new total = length(new_minimizer)
JLD2.save_object(OLD_ESTIMATES, (; minimizer = new_minimizer))
@info "done. old fit preserved at $BACKUP"
