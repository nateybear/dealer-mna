# Standalone: flatten the fitted δ vector into a DataFrame with the (dealer, vehicle,
# quarter, price, price instrument) tuple that defines each mean-utility row.
# No Snakefile rule — run as: julia --project=. scripts/demand_delta_decomposition.jl
# Writes data/demand_delta_decomposition.arrow.

using DealerMnA
import .Demand: DemandModel, n_delta
using FixedEffectModels

const MODEL_PATH = "data/model_frames.jld2"
const FIT_PATH = "data/demand_estimates.jld2"
const OUT_PATH = "data/demand_delta_decomposition.arrow"

model = DemandModel(; JLD2.load_object(MODEL_PATH)...)
fit = JLD2.load_object(FIT_PATH)
δ_hat = Float32.(fit.minimizer[1:n_delta(model)])

N = n_delta(model)
@assert length(keys(model.product_lookup.tjf)) == N

year = Vector{Int64}(undef, N)
quarter = Vector{Int64}(undef, N)
make = Vector{String}(undef, N)
model_name = Vector{String}(undef, N)
model_year = Vector{Int64}(undef, N)
trim_rank = Vector{Float32}(undef, N)
dealer_uid = Vector{UUIDs.UUID}(undef, N)
group_uid = Vector{UUIDs.UUID}(undef, N)
price = Vector{Float32}(undef, N)
price_hausman = Vector{Float32}(undef, N)
count = Vector{Int64}(undef, N)

# product_lookup.tjf maps each (t, j, f) triple to its flat Int32 product index.
# The OrderedDict iterates in insertion order, which matches the 1:N δ ordering.
for (i, (key, idx)) in enumerate(pairs(model.product_lookup.tjf))
    (t_key, j_key, f_key) = key
    yr, qtr = t_key
    mk, md, my, tr = j_key
    info = model.product_info[idx]

    year[i] = yr
    quarter[i] = qtr
    make[i] = mk
    model_name[i] = md
    model_year[i] = my
    trim_rank[i] = tr
    dealer_uid[i] = f_key
    group_uid[i] = info.g
    price[i] = info.p
    price_hausman[i] = info.p_hausman
    count[i] = info.n
end

df = DataFrame(
    delta = δ_hat,
    dealer_uid = dealer_uid,
    group_uid = group_uid,
    make = make,
    model = model_name,
    model_year = model_year,
    trim_rank = trim_rank,
    year = year,
    quarter = quarter,
    price = price,
    price_hausman = price_hausman,
    count = count,
)

df.vehicleid = string.(df.model, "_", df.model_year, "_", df.trim_rank)

transform!(groupby(df, [:year, :quarter]), nrow => :J)

regout = reg(df, @formula(delta ~ (price ~ price_hausman) + log(J) + fe(vehicleid) + fe(dealer_uid) + fe(year) + fe(quarter)), Vcov.robust(), save = :all)

first_stage = reg(df, @formula(price ~ price_hausman + log(J) + fe(vehicleid) + fe(dealer_uid) + fe(year) + fe(quarter)), Vcov.robust(), save = :fe)

JLD2.save_object("data/delta_decomposition_regout.jld2", (;regout, first_stage, df))
