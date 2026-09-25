using DealerMnA
using .Demand

using SparseArrays

import Lazy: iterated

const β = 0.95^0.25

const T_START = QuarterKey[(y, q) for y in 2017:2018 for q in 1:4]

const N_SCENARIOS = Supply.SIM_SIZE

# Must match `COORDINATED_SHIFTS` in scripts/bbl_forward_simulation.jl.
# Kept as a plain const rather than an include! to avoid pulling that script's
# top-level `using` into this module. Update both if you change one.
const N_SHIFTS = 8

increment((y, q)) =
    let new_q = (q % 4) + 1
        y_inc = q ÷ 4
        (y + y_inc, new_q)
    end

# -----------------------------------------------------------------------------
# BBLIndex: canonical row/column orderings shared across every (fe_type,
# policy) matrix pair. Every field is deterministic — derived from the
# filtered Counterfactual + usable_vks + T_START, all identical across
# sim seeds.
# -----------------------------------------------------------------------------
struct BBLIndex
    tf_rows::Vector{Tuple{QuarterKey, UUID}}
    dealer_cols::Vector{UUID}
    vehicle_cols::Vector{VehicleKey}
    quarter_cols::Vector{Int}
    tf_row_index::Dict{Tuple{QuarterKey, UUID}, Int}
    dealer_col_index::Dict{UUID, Int}
    vehicle_col_index::Dict{VehicleKey, Int}
end

function BBLIndex(dealers::AbstractVector{UUID}, usable_vks)
    tf_rows = [(t, f) for t in T_START for f in dealers]
    dealer_cols = collect(dealers)
    vehicle_cols = sort(collect(usable_vks))
    quarter_cols = [1, 2, 3, 4]
    return BBLIndex(
        tf_rows,
        dealer_cols,
        vehicle_cols,
        quarter_cols,
        Dict(k => i for (i, k) in enumerate(tf_rows)),
        Dict(k => i for (i, k) in enumerate(dealer_cols)),
        Dict(k => i for (i, k) in enumerate(vehicle_cols)),
    )
end

# Outer matrix structure:
# Rows are the different scenarios (perturbations)
# Columns are the simulations (1-100)
# Inner matrix structure:
# Unique tf combos are rows
# "basis functions" (indicator functions) are columns
const V = Vector{Float32}
const M = SparseMatrixCSC{Float32}
struct BBLMatrices
    index::BBLIndex
    R_f::Matrix{V}
    C_f::Matrix{M}
    R_t::Matrix{V}
    C_t::Matrix{M}
    R_j::Matrix{V}
    C_j::Matrix{M}
end

function BBLMatrices(idx::BBLIndex, simulations)
    ### construction helpers
    # Row 1        = observed policy
    # Rows 2..1+N_SCENARIOS       = per-product independent local perturbations (residual demand seeds)
    # Rows 2+N_SCENARIOS..end     = coordinated uniform shifts (COORDINATED_SHIFTS)
    outerM = N_SCENARIOS + N_SHIFTS + 1
    outerN = length(simulations)
    filled(T, closure) =
        let out = Array{T}(undef, outerM, outerN)
            for (m, n) in Iterators.product(1:outerM, 1:outerN)
                out[m, n] = closure()
            end
            out
        end
    newV() = filled(V, () -> zeros(Float32, length(idx.tf_rows)))
    newM(J) = filled(M, () -> spzeros(Float32, length(idx.tf_rows), J))

    ### aggregate helpers
    j_lookup((_, _, j)) = idx.vehicle_col_index[j]
    t_lookup(((_, q), _, _)) = q
    f_lookup((_, f, _)) = idx.dealer_col_index[f]

    policy_lookup(i_j, outcome) = outcome.prices[i_j], outcome.q[i_j]
    function cf_lookup(n)
        return (i_j, outcome) ->
            get.(
                (outcome.perturbation_prices, outcome.perturbation_demands),
                Ref((n, i_j)),
                0.0,
            )
    end
    # Coordinated uniform shifts share the same (n, i_j) lookup shape as the
    # per-product perturbations but read from a distinct pair of matrices.
    function coord_lookup(n)
        return (i_j, outcome) ->
            get.(
                (outcome.coordinated_prices, outcome.coordinated_demands),
                Ref((n, i_j)),
                0.0,
            )
    end

    aggregator(col_lookup, scenario_lookup) = function (tf, tup, sim)
        future_t, f, j = tup
        row = idx.tf_row_index[tf]
        col = col_lookup(tup)
        dealer_outcome = sim[(future_t, f)]
        i_j = findfirst(==(j), dealer_outcome.assortment)
        p, D = scenario_lookup(i_j, dealer_outcome)
        return (; D, p, i = (row, col))
    end

    R_f = newV()
    C_f = newM(length(idx.dealer_cols))
    R_t = newV()
    C_t = newM(4)
    R_j = newV()
    C_j = newM(length(idx.vehicle_cols))

    containers_and_lookups = (
        (f_lookup, R_f, C_f),
        (t_lookup, R_t, C_t),
        (j_lookup, R_j, C_j),
    )

    for (isim, simfile) in enumerate(simulations)
        sim = JLD2.load_object(simfile)
        for tf in idx.tf_rows
            for (lookup, Rs, Cs) in containers_and_lookups
                aggregate!(aggregator(lookup, policy_lookup), tf, sim, Rs[1, isim], Cs[1, isim])
                for iscen in 1:N_SCENARIOS
                    aggregate!(
                        aggregator(lookup, cf_lookup(iscen)),
                        tf,
                        sim,
                        Rs[iscen + 1, isim],
                        Cs[iscen + 1, isim],
                    )
                end
                for ishift in 1:N_SHIFTS
                    aggregate!(
                        aggregator(lookup, coord_lookup(ishift)),
                        tf,
                        sim,
                        Rs[N_SCENARIOS + 1 + ishift, isim],
                        Cs[N_SCENARIOS + 1 + ishift, isim],
                    )
                end
            end
        end
    end
    return BBLMatrices(idx, R_f, C_f, R_t, C_t, R_j, C_j)
end

# for a given (t,f) tuple, walk forward a year and add it's contribution to our average W (from BBL)
# fun should take a (t, f, j) tuple and a sim object and return a named tuple, (D realized demand, p chosen price, i index in C)
# Rows in C correspond to positions in R, just a unique index for each (t,f) combination
# Columns in C correspond to positions for different fixed effect indicators
function aggregate!(fun, tf, sim, R::AbstractVector, C::SparseMatrixCSC)
    (t, f) = tf
    for (fwd, new_t) in enumerate(first(iterated(increment, t), 5))
        fwd -= 1 # number of periods ahead we are
        for j in sim[(new_t, f)].assortment
            (; D, p, i) = fun(tf, (new_t, f, j), sim)
            R[i[1]] += (β^fwd) * D * p  # βᵗ D(p) * p
            C[i...] -= (β^fwd) * D # - βᵗ D(p) (and a slot for a component of mc_{jft})
        end
    end
    return nothing
end

function (@main)(_)
    snakemake!()
    with_logging() do
        # ---------------------------------------------------------------------
        # Load — subset of bbl_forward_simulation.jl:22-38 needed to reproduce
        # the eviction + usable_vks passes. No compliance / sales_targets /
        # price_table / residual_demand here — those drive sim-time draws but
        # don't affect which (t, f, j) survive the two eviction passes.
        # ---------------------------------------------------------------------
        model = Demand.DemandModel(;
            JLD2.load_object(snakemake.input["model_frames"])...
        )
        params = JLD2.load_object(snakemake.input["demand_estimates"]).minimizer
        cf = Demand.Counterfactual(
            model, params; decomp_path = snakemake.input["delta_decomp"]
        )
        price_densities = JLD2.load_object(snakemake.input["price_densities"])
        state_transition = JLD2.load_object(snakemake.input["state_transition"])

        # ---------------------------------------------------------------------
        # First eviction — mirrors bbl_forward_simulation.jl:39-74. Products
        # whose dealer_uid or vehicleid has no identifiable FE in regout are
        # removed so sweep_δ! / pred_regout doesn't hit `missing` fe values.
        # ---------------------------------------------------------------------
        fe_df = cf.regout.fe
        identifiable_vehicleids = Set{String}(
            row.vehicleid for row in eachrow(fe_df) if !ismissing(row.fe_vehicleid)
        )
        identifiable_dealers = Set{UUID}(
            row.dealer_uid for row in eachrow(fe_df) if !ismissing(row.fe_dealer_uid)
        )
        vk_to_vehicleid(vk) = string(vk[2], "_", vk[3], "_", Float32(vk[4]))

        for (tjf_key, _) in collect(pairs(cf.model.product_lookup.tjf))
            (t_key, j_key, f_key) = tjf_key
            bad_dealer = f_key ∉ identifiable_dealers
            bad_vk = vk_to_vehicleid(j_key) ∉ identifiable_vehicleids
            (bad_dealer || bad_vk) || continue
            apply!(
                cf, Demand.ModelDelete(vehicle = j_key, quarter = t_key, dealer_uid = f_key)
            )
        end
        Supply.invalidate_tf_to_j_lookup!()
        # Ackerberg-Rysman correction: J_t shrank, so pred_regout's log(J)
        # covariate makes every stored δ stale. Sweep every quarter before any
        # downstream expected_demand / residual_demand read runs.
        for t_sweep in unique(k[1] for k in keys(cf.model.product_lookup.tjf))
            Demand.sweep_δ!(cf, t_sweep)
        end
        Demand.refresh!(cf)

        # Snapshot state_lookup HERE — mirrors bbl_forward_simulation.jl:79-80,
        # captured after first eviction but before the usable_vks eviction so
        # that `dealers` below matches the sim's `dealers` set exactly.
        intermed = Supply.tf_to_j_lookup(model)
        state_lookup = Dict(k => i for (i, k) in enumerate(keys(intermed)))

        # ---------------------------------------------------------------------
        # usable_vks — mirrors bbl_forward_simulation.jl:167-183. Simplified
        # because keys(price_mean_boosters) == keys(price_var_boosters) ==
        # keys(price_densities) (the bootup loop assigns both unconditionally),
        # and likewise for state_transition's int/own boosters.
        # ---------------------------------------------------------------------
        T::Vector{QuarterKey} = [(y, q) for y in 2017:2019 for q in 1:4]
        max_model_year = maximum(y for (y, _) in T)
        usable_vks = intersect(
            Set(keys(price_densities)),
            Set(keys(state_transition)),
            Set(keys(model.product_lookup.j)),
        )
        filter!(vk -> vk[3] <= max_model_year, usable_vks)

        # ---------------------------------------------------------------------
        # Second eviction — mirrors bbl_forward_simulation.jl:193-205. Persist-
        # ently removes non-usable VKs so the DemandModel's live universe
        # matches the sim's sampling universe.
        # ---------------------------------------------------------------------
        for (tjf_key, _) in collect(pairs(cf.model.product_lookup.tjf))
            (t_key, j_key, f_key) = tjf_key
            j_key ∈ usable_vks && continue
            apply!(
                cf, Demand.ModelDelete(vehicle = j_key, quarter = t_key, dealer_uid = f_key)
            )
        end
        Supply.invalidate_tf_to_j_lookup!()
        # AR correction — see first-eviction sweep above.
        for t_sweep in unique(k[1] for k in keys(cf.model.product_lookup.tjf))
            Demand.sweep_δ!(cf, t_sweep)
        end
        Demand.refresh!(cf)

        # ---------------------------------------------------------------------
        # dealers — mirrors bbl_forward_simulation.jl:207. Read off the
        # first-eviction state_lookup so this matches the sim byte-for-byte.
        # ---------------------------------------------------------------------
        t₀ = T[1]
        dealers = sort(UUID[f for ((t, f), _) in state_lookup if t == t₀])
        @info "BBLIndex derivation" n_dealers = length(dealers) n_usable_vks = length(
            usable_vks
        )

        idx = BBLIndex(dealers, usable_vks)

        # Fail loud if the sim is missing any (t, f) the forward walk in
        # `aggregate!` will read. Check the union of walk_forward(tf, 5) —
        # not just tf_rows — since aggregate! looks up sim[(new_t, f)] for
        # new_t past T_START.
        sim0 = JLD2.load_object(snakemake.input["simulations"][1])
        expected = Set(
            (new_t, f) for (t, f) in idx.tf_rows for
            new_t in first(iterated(increment, t), 5)
        )
        missing_keys = [k for k in expected if !haskey(sim0, k)]
        if !isempty(missing_keys)
            @error "sim missing (t, f) that aggregate! will read" n_missing = length(
                missing_keys
            ) example = missing_keys[1]
            error("Regenerate sim files: snakemake -F data/bbl_simulation/seed_{1..100}.jld2")
        end

        JLD2.save_object(snakemake.output[1], BBLMatrices(idx, snakemake.input["simulations"]))
    end
    return nothing
end
