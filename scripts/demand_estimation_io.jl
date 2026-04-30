###############################################################################
# Sample: estimate the dealer-car demand model via IndustrialOrganization.jl
#
# This script demonstrates one full pass through:
#   1. munging our `model_frames` data into the IO package's per-market schema
#   2. defining a custom AbstractDesign that carries per-market distance matrices
#   3. overloading `IndirectUtility!` to add −exp(η₁ + η₂·D[j,i]) to consumer
#      utility on the micro path
#   4. running `Estimate!` and pulling results via `Summarize!`
#
# Caveats (read these before relying on the output):
#
#   • The IO package's parameter vector θ has slots only for interactions (θᶻ)
#     and random coefficients (θᵛ). To get *two* free distance parameters
#     (η₁, η₂) into the optimizer's parameter space without patching the
#     package, this script co-opts two "fake" random-coefficient slots: two
#     product characteristics with identically-zero columns in `xvcharacteristics`
#     and zero V-draws, so the standard linear-utility path adds 0 for them.
#     Our IndirectUtility! override pulls those θᵛ entries out and uses them
#     as η in −exp(η₁ + η₂·D). When the package author lands proper support
#     for arbitrary indirect utilities, swap to that.
#
#   • Joint estimation of η alongside δ requires the gradients/Hessians of the
#     micro objective with respect to those slots to be correct. That fix is
#     pending from the package author; we proceed assuming it lands.
#
#   • A "market" here is a single quarter. Products within a market are
#     (dealer × model-trim) combinations available statewide that quarter.
#     Consumers are individual purchasers; their demographics are the
#     duplicated zip-level demographics (one row per actual purchase).
#
# Run interactively:
#   julia --project=. scripts/demand_estimation_io.jl
#
# Or under snakemake (will need a rule mirroring data_demand_estimates).
###############################################################################

using DealerMnA
using DataFrames
using LinearAlgebra
using Random
using Dates
using StatsBase: mean
using AxisKeys: KeyedArray
using DataStructures: SortedDict

import IndustrialOrganization as IO
import IndustrialOrganization:
    AbstractDesign,
    DiscreteChoiceRandomCoefficientsLinearIndirectUtilityDemandModel,
    Market💼,
    Estimate!,
    Summarize!

Random.seed!(78704)

snakemake!()  # no-op if running interactively

const RUNNING_UNDER_SNAKEMAKE = !isnothing(snakemake)

# ──────────────────────────────────────────────────────────────────────────────
# 1. CUSTOM DESIGN AND INDIRECTUTILITY OVERLOAD
# ──────────────────────────────────────────────────────────────────────────────
# DistanceDesign carries one J × I distance matrix per market, keyed by the IO
# market name (we use the marketname `"q$year-$quarter"`). The design tracks
# the *current* market via a `Ref{String}` that we set in a thin override of
# ChoiceProbabilities!, which is the only call site that knows the market.

struct DistanceDesign{T<:Real} <: AbstractDesign
    "marketname → (J+1) × I distance matrix; outside good (last row) gets 0."
    D::Dict{String, Matrix{T}}
    "current marketname; mutated by ChoiceProbabilities! before IndirectUtility!"
    current::Ref{String}
end
DistanceDesign(D::Dict{String, Matrix{T}}) where {T<:Real} =
    DistanceDesign{T}(D, Ref(""))

# Override ChoiceProbabilities! on Market💼 with our design, only to set
# `design.current[]` and then defer to the standard linear path. We need to
# hit the package's internal call site, which lives in
# src/optim/randomcoefficients/choiceprobabilities/micro.jl.
function IO.ChoiceProbabilities!(
    💼::Market💼{𝒯,
        DiscreteChoiceRandomCoefficientsLinearIndirectUtilityDemandModel,
        ê, ℹ, 🚠, 👾, 🪂, DistanceDesign{𝒯}, 🙏, 📖, 🏆},
    σᵇ::AbstractArray{𝒯, 3},
) where {𝒯<:Real, ê, ℹ, 🚠, 👾, 🪂, 🙏, 📖, 🏆}

    💼.design.current[] = 💼.ℹ.marketname
    # Delegate to the package implementation by calling IndirectUtility! +
    # softmax exactly the way micro.jl does. See micro.jl for the original.
    @raw 👗, ℹ_, 🪂_, design_ = 💼
    Z, Xᶻ, Xᵛ, Vᵐ = ℹ_.Z, ℹ_.Xᶻ, ℹ_.Xᵛ, ℹ_.Vᵐ
    θᶻ = IO.getθᶻ(🪂_, true)
    θᵛ = IO.getθᵛ(🪂_, true)

    IO.IndirectUtility!(👗, σᵇ, Z, Xᶻ, Xᵛ, Vᵐ, θᶻ, θᵛ, design_)

    # softmax over products for each (draw, consumer)
    @inbounds for i in axes(σᵇ, 3), r in axes(σᵇ, 2)
        m = σᵇ[1, r, i]
        for j in axes(σᵇ, 1)
            v = σᵇ[j, r, i]
            v > m && (m = v)
        end
        s = zero(𝒯)
        for j in axes(σᵇ, 1)
            σᵇ[j, r, i] = exp(σᵇ[j, r, i] - m)
            s += σᵇ[j, r, i]
        end
        s = inv(s)
        for j in axes(σᵇ, 1)
            σᵇ[j, r, i] *= s
        end
    end
    return nothing
end

# Override IndirectUtility! to add the distance term. We start from the
# package's linear-model formula:
#   σ[j,r,i] = (Xᶻʷ Z')[j,i] + (Xᵛʷ V')[j,r]
# and add the distance disutility −exp(η₁ + η₂·D[j,i]) for every (j,r,i).
# η are the *last two* entries of θᵛ by convention here (see comment block at
# top about the fake-RC trick).
function IO.IndirectUtility!(
    ::DiscreteChoiceRandomCoefficientsLinearIndirectUtilityDemandModel,
    σ::AbstractArray{𝒯, 3},
    Z::AbstractMatrix{𝒯}, Xᶻ::AbstractMatrix{𝒯}, Xᵛ::AbstractMatrix{𝒯},
    V::AbstractMatrix{𝒯},
    θᶻ::AbstractVector{𝒯}, θᵛ::AbstractVector{𝒯},
    design::DistanceDesign{𝒯},
) where {𝒯<:Real}

    # Standard linear contribution (verbatim from micro.jl).
    Xᶻʷ = Xᶻ .* reshape(θᶻ, 1, :)
    Xᵛʷ = Xᵛ .* reshape(θᵛ, 1, :)
    Uᶻ = Xᶻʷ * transpose(Z)
    Uᵛ = Xᵛʷ * transpose(V)
    σ .= reshape(Uᶻ, size(Uᶻ, 1), 1, size(Uᶻ, 2)) .+
         reshape(Uᵛ, size(Uᵛ, 1), size(Uᵛ, 2), 1)

    # Distance contribution — pull η out of the *last two* entries of θᵛ.
    # `D` is (J+1) × I for the current market; its last row is 0 for the
    # outside good.
    D = design.D[design.current[]]
    η₁, η₂ = θᵛ[end - 1], θᵛ[end]
    @inbounds for i in axes(σ, 3)
        for j in axes(σ, 1)
            disutility = exp(η₁ + η₂ * D[j, i])
            for r in axes(σ, 2)
                σ[j, r, i] -= disutility
            end
        end
    end
    return σ
end

# ──────────────────────────────────────────────────────────────────────────────
# 2. LOAD OUR MODEL FRAMES
# ──────────────────────────────────────────────────────────────────────────────
# We re-use Demand.load_model_frames-shaped data, since it already aggregates
# (zip × quarter × product) counts and (quarter × product) mean prices. If
# `data/model_frames.jld2` already exists we load that; otherwise rebuild.

const MF_PATH = if RUNNING_UNDER_SNAKEMAKE
    snakemake.input["model_frames"]
else
    DealerMnA.here("data/model_frames.jld2")
end

@info "Loading model frames from $MF_PATH"
mf = JLD2.load_object(MF_PATH)
# `mf` has fields: distances, product_info, product_lookup, agent_data,
# counts, counts_lookup, ownership, ownership_lookup. See
# src/demand/loading.jl.

# ──────────────────────────────────────────────────────────────────────────────
# 3. RESHAPE INTO IO'S PER-MARKET SCHEMA
# ──────────────────────────────────────────────────────────────────────────────
# Plan:
#   • markets = unique quarters t = (year, quarter)
#   • per-market product list = unique (j, f) appearing in product_info[t,:,:]
#   • outside-good row appended (`outside`)
#   • shares s = (count / market-size N), with N = sum of zip populations
#   • consumer rows = expand counts[z, t, j, f] into individual purchase rows
#     (one row per purchase) keyed by zip
#   • demographics duplicated from agent_data[zip] across all consumer rows
#   • distance D[j, i] = mf.distances[zip(i), f(j)] (last row = 0)

const T_VALUES = sort(collect(keys(mf.product_lookup.t)))
const ZIP_VALUES = sort(collect(keys(mf.agent_data)))

marketname(t::Tuple{Int, Int}) = "q$(t[1])-$(t[2])"

# All product columns IO needs — names must be valid Julia identifiers.
# Demographic order matches `Demand.DEMOGRAPHIC_ATTRIBUTES`.
const DEMO_COLS = [:median_earnings_inv, :urban_pct, :black_pct, :hispanic_pct]
const PROD_COLS = [:is_coupe, :is_suv, :is_pickup, :is_sedan]
const PRICE_COL = :price
const HAUSMAN_COL = :p_hausman          # used as a Z (instrument) on the moment side
# Two "fake" random-coefficient slots that we co-opt as η₁, η₂. Both are
# zero-valued so the standard linear path contributes 0 for them.
const ETA_COLS = [:eta1_dummy, :eta2_dummy]

# Per-market frames
products_dfs = DataFrame[]
consumers_dfs = DataFrame[]
marketsize_rows = NamedTuple[]
distance_dict = Dict{String, Matrix{Float64}}()
draws_dfs = DataFrame[]

# We need the inverse maps: from CartesianIndex into product_info → (t,j,f)
product_keys_t = collect(keys(mf.product_lookup.t))
product_keys_j = collect(keys(mf.product_lookup.j))
product_keys_f = collect(keys(mf.product_lookup.f))

# Fast access to consumer counts per zip in each quarter.
counts_keys_z = collect(keys(mf.counts_lookup.z))
zip_index = Dict(z => i for (i, z) in enumerate(counts_keys_z))

for t in T_VALUES
    mname = marketname(t)
    @info "Building market $mname"

    # ── products list for market t ─────────────────────────────────────────
    # Iterate over product_info entries with first index == time index of t.
    t_idx = findfirst(==(t), product_keys_t)
    prod_indices = collect(mf.product_lookup.t[t])  # CartesianIndex{3}, in (t,j,f) space
    prod_indices = sort(prod_indices)
    prod_rows = NamedTuple[]
    for ci in prod_indices
        (it, ij, ifc) = Tuple(ci)
        info = mf.product_info[ci]
        j_key = product_keys_j[ij]
        f_key = product_keys_f[ifc]
        push!(prod_rows, (
            product = "$(j_key[1])|$(j_key[2])|$(j_key[3])|$(j_key[4])|$(f_key)",
            f_uid = f_key,
            is_coupe = Float64(info.is_coupe),
            is_suv = Float64(info.is_suv),
            is_pickup = Float64(info.is_pickup),
            is_sedan = Float64(info.is_sedan),
            price = Float64(info.p),
            p_hausman = Float64(info.p_hausman),
            eta1_dummy = 0.0,
            eta2_dummy = 0.0,
            n = Int64(info.n),
        ))
    end
    # outside good row — name must match the `:outsidegood` symbol below.
    push!(prod_rows, (
        product = "outside",
        f_uid = DealerMnA.SINGLE_OPERATOR_GID,  # placeholder
        is_coupe = 0.0, is_suv = 0.0, is_pickup = 0.0, is_sedan = 0.0,
        price = 0.0, p_hausman = 0.0,
        eta1_dummy = 0.0, eta2_dummy = 0.0,
        n = 0,
    ))
    products_df = DataFrame(prod_rows)
    products_df.market .= mname

    # ── market size N ──────────────────────────────────────────────────────
    # Approximation: total potential market size = sum over zips of M (= pop/48).
    N = sum(z -> mf.agent_data[z].M, ZIP_VALUES)

    # ── consumer rows: expand counts[z, t, j, f] into individual purchases ──
    # And build distance matrix in same product order.
    cons_rows = NamedTuple[]
    f_to_jrow = Dict(row.f_uid => i for (i, row) in enumerate(eachrow(products_df)))
    n_products = nrow(products_df)

    # We need the product index for each (j, f) pair — the f_to_jrow map only
    # works when there's a single product per dealer per market, which is NOT
    # our case. Build a (j_key, f_uid) → row index map instead.
    jf_to_row = Dict{Tuple{Tuple{String,String,Int64,Float32}, Any}, Int}()
    for (i, row) in enumerate(eachrow(products_df))
        if row.product == "outside"
            continue
        end
        parts = split(row.product, "|")
        jkey = (parts[1], parts[2], parse(Int64, parts[3]), parse(Float32, parts[4]))
        jf_to_row[(jkey, row.f_uid)] = i
    end

    consumer_id = 0
    consumer_zip = String[]
    for z in ZIP_VALUES
        z_idx = zip_index[z]
        # Iterate over consumer purchases in this zip × this quarter.
        # counts[z, t, j, f] is sparse; we walk product_info entries that have
        # any counts in this (z, t).
        for ci in prod_indices
            (it, ij, ifc) = Tuple(ci)
            j_key = product_keys_j[ij]
            f_key = product_keys_f[ifc]
            # mf.counts is keyed (z, t, j, f) — same j and f indices as product_info
            count = try
                mf.counts[z_idx, it, ij, ifc]
            catch
                0
            end
            count = Int(count)
            for _ in 1:count
                consumer_id += 1
                push!(consumer_zip, z)
                demos = mf.agent_data[z]
                push!(cons_rows, (
                    consumer = consumer_id,
                    choice = jf_to_row[(j_key, f_key)],
                    median_earnings_inv = Float64(demos.median_earnings_inv),
                    urban_pct = Float64(demos.urban_pct),
                    black_pct = Float64(demos.black_pct),
                    hispanic_pct = Float64(demos.hispanic_pct),
                ))
            end
        end
    end
    if isempty(cons_rows)
        @warn "No consumer rows for market $mname; skipping"
        continue
    end
    consumers_df = DataFrame(cons_rows)
    consumers_df.market .= mname

    # ── distance matrix: (J+1) × I, last row = 0 (outside) ─────────────────
    n_cons = length(consumer_zip)
    D = zeros(Float64, n_products, n_cons)
    for (i, z) in enumerate(consumer_zip)
        for (j, row) in enumerate(eachrow(products_df))
            if row.product == "outside"
                D[j, i] = 0.0
            else
                D[j, i] = Float64(mf.distances[z, row.f_uid])
            end
        end
    end
    distance_dict[mname] = D

    # ── per-market integration draws ───────────────────────────────────────
    # Random-coefficient draws are required by IO; for our two fake RC slots
    # with zero coefficients, we just feed zeros — the integral collapses.
    R = 50  # number of quadrature/MC nodes for micro integration
    draws_rows = NamedTuple[]
    for r in 1:R
        push!(draws_rows, (
            eta1_dummy = 0.0,
            eta2_dummy = 0.0,
            weight = 1.0 / R,
        ))
    end
    draws_df = DataFrame(draws_rows)
    draws_df.market .= mname

    push!(products_dfs, products_df)
    push!(consumers_dfs, consumers_df)
    push!(marketsize_rows, (N = Float64(N), market = mname))
    push!(draws_dfs, draws_df)
end

products_all = reduce(vcat, products_dfs)
consumers_all = reduce(vcat, consumers_dfs)
marketsizes_all = DataFrame(marketsize_rows)
draws_all = reduce(vcat, draws_dfs)

# Compute observed shares from counts and N for each market, append to products.
transform!(
    groupby(products_all, :market),
    [:n] => (n -> n ./ max(sum(n), 1)) => :share,
)

# ──────────────────────────────────────────────────────────────────────────────
# 4. BUILD THE IO INPUT DICT AND ESTIMATE
# ──────────────────────────────────────────────────────────────────────────────
# Interactions (xzcharacteristics × demographics): each demographic interacts
# with all four body-class dummies plus price. That's 4 × 5 = 20 interactions,
# matching Demand.n_demo_product_params + Demand.n_demo_price_params.
xz_chars = Symbol[]
demos = Symbol[]
for d in DEMO_COLS, x in vcat(PROD_COLS, [PRICE_COL])
    push!(xz_chars, x)
    push!(demos, d)
end

# Random-coef slots: just our two fake η columns.
xv_chars = ETA_COLS

# Linear regression of δ on these to recover β (the price coef and FE)
regressors = vcat(PROD_COLS, [PRICE_COL])
# Instruments: same characteristics plus the Hausman price (excluded).
instruments = vcat(PROD_COLS, [PRICE_COL, HAUSMAN_COL])

D = SortedDict{Symbol, Any}(
    :model => :rclindem,
    :productdata => products_all,
    :consumerdata => consumers_all,
    :marketsizedata => marketsizes_all,
    :drawsdata => draws_all,
    :choices => :choice,
    :shares => :share,
    :markets => :market,
    :marketsizes => :N,
    :products => :product,
    :outsidegood => "outside",
    :demographics => demos,
    :xzcharacteristics => xz_chars,
    :xvcharacteristics => xv_chars,
    :regressors => regressors,
    :instruments => instruments,
    :design => DistanceDesign(distance_dict),
    :𝒯 => Float64,
)

@info "Calling IndustrialOrganization.Estimate!"
briefcase = Estimate!(D)

@info "Calling Summarize!"
results = Summarize!(briefcase; pricevar = PRICE_COL)

# ──────────────────────────────────────────────────────────────────────────────
# 5. EXTRACT AND SAVE
# ──────────────────────────────────────────────────────────────────────────────
@info "θ" briefcase.parameters.θ
@info "regression results" results."regression results"

# η₁, η₂ are the last two entries of θᵛ; pull and report explicitly.
θᵛ = IO.getθᵛ(briefcase.parameters)
η₁, η₂ = θᵛ[end - 1], θᵛ[end]
@info "Distance disutility parameters" η₁ η₂

if RUNNING_UNDER_SNAKEMAKE
    output_path = snakemake.output[1]
    JLD2.save_object(output_path, (briefcase = briefcase, results = results))

    # LaTeX dump of the regression table, alongside the JLD2.
    p, _ext = splitext(output_path)
    open("$(p)_regression.tex", "w") do io
        show(io, IO.LaTeX, results."regression results")
    end
end
