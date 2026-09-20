using DealerMnA
using DataFrames
using Statistics: mean, quantile

function (@main)(_)
    snakemake!()
    with_logging() do
        fits = JLD2.load_object(snakemake.input["compliance_fits_final"])
        state_matrix = JLD2.load_object(snakemake.input["residual_demand"])
        @info "Loaded" n_groups = length(fits) n_state_rows = size(state_matrix, 1) n_state_cols = size(
            state_matrix, 2
        )

        detailed = DataFrame(
            model = String[], model_year = Int64[], extra = String[],
            target = String[],
            tree_idx = Int[], node_id = Int[], depth = Int[],
            feature = Int[], threshold = Float64[], missing_left = Bool[],
            n_missing = Int[], n_below = Int[], n_above = Int[],
            min_nonmiss = Float64[], max_nonmiss = Float64[],
            threshold_trivial = Bool[],
        )

        for ((model, model_year, extra), (pbase_bytes, rho_bytes)) in fits
            for (target, bytes) in (("p_base", pbase_bytes), ("rho", rho_bytes))
                isempty(bytes) && continue
                splits = Supply.analyze_tree_splits(bytes, state_matrix)
                splits.model .= model
                splits.model_year .= model_year
                splits.extra .= extra
                splits.target .= target
                append!(detailed, splits)
            end
        end
        @info "Analyzed booster trees" n_splits = nrow(detailed)

        Arrow.write(snakemake.output["detailed"], detailed)
        @info "Wrote detailed per-split table" path = snakemake.output["detailed"]

        # Per-booster summary: what fraction of splits are trivial, how many
        # unique features get used, mean fraction of rows missing at each split.
        n_rows = size(state_matrix, 1)
        summary = @> detailed groupby([:model, :model_year, :extra, :target]) combine(
            nrow => :n_splits,
            :threshold_trivial => mean => :frac_trivial,
            :feature => (fs -> length(unique(fs))) => :n_unique_features,
            :n_missing => (m -> mean(m) / n_rows) => :mean_frac_missing,
            :depth => mean => :mean_depth,
            :depth => maximum => :max_depth,
        )
        sort!(summary, [:target, :frac_trivial], rev = [false, true])
        Arrow.write(snakemake.output["summary"], summary)
        @info "Per-booster summary" nrows = nrow(summary)

        # Top-line reduction across all groups.
        for t in ("p_base", "rho")
            sub = subset(summary, :target => ByRow(==(t)))
            isempty(sub) && continue
            @info "Overall by target" target = t n_boosters = nrow(sub) mean_frac_trivial = mean(
                sub.frac_trivial
            ) q_frac_trivial = quantile(sub.frac_trivial, [0.1, 0.5, 0.9]) mean_frac_missing = mean(
                sub.mean_frac_missing
            ) mean_n_unique_features = mean(sub.n_unique_features)
        end
    end
    return nothing
end
