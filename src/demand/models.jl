export DemandModel

struct DemandModel
    # Standardization scales (divide raw covariates by these to enter the model)
    demo_scales::Vector{Float32}
    price_scale::Float32
    dist_scale::Float32

    # Raw load-time data
    distances::NamedMatrix{Float32}                   # zips × dealers (standardized)
    product_info::Dict{Int32, ProductInfo}            # δ_idx → info (stable keys across tweaks)
    product_lookup::ProductLookup
    agent_data::Dict{String, AgentInfo}
    zips::Vector{String}                              # column order for D_dense/dist_blocks/counts_blocks

    # Precomputed dense layout
    N::Int
    Z::Int
    K::Int
    Dn::Int
    n_blocks::Int
    block_keys::Vector{QuarterKey}                    # sorted (year, quarter)
    block_param_idxs::Vector{Vector{Int32}}           # δ indices in each block
    X_blocks::Vector{Matrix{Float32}}                 # (K+1) × N_bi
    dist_blocks::Vector{Matrix{Float32}}              # N_bi × Z
    counts_blocks::Vector{SparseMatrixCSC{Int32, Int32}}  # N_bi × Z
    t_obs::Matrix{Int32}                              # n_blocks × Z
    D_dense::Matrix{Float32}                          # Dn × Z
    M_zs::Vector{Float32}                             # Z
    tjf_to_block_k::Dict{Int32, Tuple{Int32, Int32}}  # δ_idx → (bi, k_local)
    inv_N::Float32
    # Tweak-time state: physical slots per block that were freed by ModelDelete and
    # are eligible for reuse by ModelAdd. Empty at load time; grows as CF tweaks are
    # applied. Not persisted — always defaults to empty on kwarg-constructor load.
    free_slots::Vector{Vector{Int}}
end

# Kwarg constructor — splats a NamedTuple loaded from data/model_frames.jld2.
# `free_slots` is optional and defaults to an empty free list per block so that
# saved JLD2 frames (which don't carry the field) continue to work.
function DemandModel(;
    demo_scales, price_scale, dist_scale, distances, product_info, product_lookup,
    agent_data, zips, N, Z, K, Dn, n_blocks, block_keys, block_param_idxs,
    X_blocks, dist_blocks, counts_blocks, t_obs, D_dense, M_zs, tjf_to_block_k, inv_N,
    free_slots = [Int[] for _ in 1:n_blocks],
)
    return DemandModel(
        demo_scales, price_scale, dist_scale, distances, product_info, product_lookup,
        agent_data, zips, N, Z, K, Dn, n_blocks, block_keys, block_param_idxs,
        X_blocks, dist_blocks, counts_blocks, t_obs, D_dense, M_zs, tjf_to_block_k, inv_N,
        free_slots,
    )
end

# Number of currently-live products in block `bi` (excludes ModelDelete-masked slots).
n_live(model::DemandModel, bi::Integer) =
    length(model.block_param_idxs[bi]) - length(model.free_slots[bi])

n_delta(model::DemandModel) = model.N
n_demographics(_) = length(DEMOGRAPHIC_ATTRIBUTES)
n_product_attributes(_) = length(PRODUCT_ATTRIBUTES)
n_demo_product_params(x) = n_demographics(x) * n_product_attributes(x)
n_demo_price_params(x) = n_demographics(x)
n_dist_params(_) = 2

const ORDER = [
    n_delta,
    n_demo_product_params,
    n_demo_price_params,
    n_dist_params,
]

total_n_params(model::DemandModel) = sum(f(model) for f in ORDER)

macro indexof(name, params, model)
    fname = Symbol("n_$(string(name))")
    quote
        i = findfirst(==($fname), ORDER)
        n_before = mapreduce(+, first(ORDER, i - 1), init = 0) do f
            f($(esc(model)))
        end
        n_current = $fname($(esc(model)))
        n_before .+ (1:n_current)
    end
end

macro pull(name, params, model)
    quote
        i = @indexof($name, $(esc(params)), $(esc(model)))
        @view $(esc(params))[i]
    end
end

function initial_parameters(model::DemandModel)
    out = zeros(Float32, total_n_params(model))

    # δ targets P(outside) ≈ 0.99 at η=-4, J≈1500.
    out[@indexof delta out model] .= -10.0f0

    # Distance disutility exp(η₁ + η₂·dist'); η₁=-4 → exp(-4)≈0.018 at d=0.
    # η₂=1.0 since dist is standardized to ~1 std-dev units.
    out[@indexof dist_params out model] .= Float32[-4.0f0, 1.0f0]

    # Πᵦ (4 product types × 4 demographics, demographics fastest):
    # [coupe=0.1, suv=0.5, pickup=1.0, sedan=0.25] replicated across all demographics.
    out[@indexof demo_product_params out model] .= repeat(
        Float32[0.1f0, 0.5f0, 1.0f0, 0.25f0], inner = n_demographics(model)
    )

    # Πp (4 demographics): uniform -0.5; after standardization Πp[d]·D'_z[d]·p' is O(1).
    out[@indexof demo_price_params out model] .= -0.5f0

    return out
end
