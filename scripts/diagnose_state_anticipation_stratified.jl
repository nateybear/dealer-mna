using DealerMnA
using DataFrames
using Statistics: mean

# Flat index in the column-major J × (J+1) residual-demand matrix →
# :intercept (column 1) / :own (diagonal, c = r+1) / :cross (everywhere else).
function classify_cell(flat_idx::Int, J::Int)
    r = ((flat_idx - 1) % J) + 1
    c = ((flat_idx - 1) ÷ J) + 1
    if c == 1
        return :intercept
    elseif c == r + 1
        return :own
    else
        return :cross
    end
end

function (@main)(_)
    snakemake!()
    with_logging() do
        # (year, quarter, class) → [sum η, sum η², count]
        by_quarter = Dict{Tuple{Int, Int, Symbol}, Vector{Float64}}()
        # class → [sum η, sum η², count]
        overall = Dict{Symbol, Vector{Float64}}()

        n_records = 0
        n_seed_skipped = 0
        n_empty_skipped = 0

        for path in snakemake.input["simulations"]
            outputs = JLD2.load_object(path)
            for ((t, _f), nt) in outputs
                # Seed period: no state_expected populated.
                if isempty(nt.state_expected)
                    n_seed_skipped += 1
                    continue
                end
                # Empty assortment: state_actual is carried over from state_prev,
                # not a fresh residual_demand fit — no shock to inspect.
                if isempty(nt.assortment)
                    n_empty_skipped += 1
                    continue
                end
                n_records += 1

                L = length(nt.state_expected)
                @assert length(nt.state_actual) == L
                # J*(J+1) = L → J = (-1 + √(1 + 4L)) / 2
                J = round(Int, (-1 + sqrt(1 + 4L)) / 2)
                @assert J * (J + 1) == L "state vector length $L not J*(J+1)"

                y, q = t
                for i in 1:L
                    e = nt.state_expected[i]
                    a = nt.state_actual[i]
                    (isnan(e) || isnan(a)) && continue
                    η = Float64(a) - Float64(e)
                    cls = classify_cell(i, J)

                    tk = (y, q, cls)
                    acc = get!(by_quarter, tk, [0.0, 0.0, 0.0])
                    acc[1] += η
                    acc[2] += η * η
                    acc[3] += 1

                    oacc = get!(overall, cls, [0.0, 0.0, 0.0])
                    oacc[1] += η
                    oacc[2] += η * η
                    oacc[3] += 1
                end
            end
        end

        @info "Scanned forward-sim outputs" n_records n_seed_skipped n_empty_skipped n_seeds =
            length(snakemake.input["simulations"])

        function summarize(acc)
            s, sq, n = acc
            m = s / n
            var = sq / n - m^2
            return m, sqrt(max(0.0, var)), Int(n)
        end

        table = DataFrame(
            year = Int[], quarter = Int[], class = String[],
            mean_eta = Float64[], std_eta = Float64[], se_mean = Float64[],
            n_cells = Int[],
        )
        for ((y, q, cls), acc) in by_quarter
            m, sd, n = summarize(acc)
            push!(
                table,
                (y, q, string(cls), m, sd, sd / sqrt(n), n),
            )
        end
        sort!(table, [:class, :year, :quarter])
        Arrow.write(snakemake.output["diagnostic"], table)
        @info "Wrote per-quarter × class table" path = snakemake.output["diagnostic"] nrows = nrow(
            table,
        )

        # Top-line: overall mean η per cell class. The "correct anticipation +
        # mean-zero shock" model predicts |mean_eta| ≲ 1 SE.
        for cls in (:intercept, :own, :cross)
            haskey(overall, cls) || continue
            m, sd, n = summarize(overall[cls])
            se = sd / sqrt(n)
            @info "Overall shock" class = cls mean_eta = m std_eta = sd se_mean = se n_cells =
                n z_score = m / se
        end
    end
    return nothing
end
