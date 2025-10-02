abstract type AbstractCoefType end
abstract type AbstractDemandModel end

struct RandomCoefType <: AbstractCoefType
    draws::Vector{Float64} # S-vector
    grid::Matrix{Float64} # S x (J + 1), last column is outside good
    new(S::Int64, J::Int64) = new(rand(Normal(0, 1), S), zeros(S, J + 1)) 
end

struct FixedCoefType <: AbstractCoefType
    draws::Vector{Float64} # 1-vector
    grid::Matrix{Float64} # 1 x (J + 1), last column is outside good
    new(::Int64, J::Int64) = new(zeros(1), zeros(1, J + 1))
    new(J::Int64) = new(1, J)
end

struct AllTrimsModel{T <: AbstractCoefType}
    u::T
    distances::NamedMatrix{Float64} # zips in rows, dealers in columns
    product_data::DataFrame # (trim, quarter, dealer, model) assume sorted
    agent_data::DataFrame # demographics for each zip code
    counts::DataFrame
end

function AllTrimsModel{T}(u::T) where {T <: AbstractCoefType}
    

    # Now need to fill all combinations of trims

    return AllTrimsModel(u, distances, product_data, agent_data, counts)
end

const PRODUCT_ATTRIBUTES = [:suv, :truck, :electric]
const DEMOGRAPHIC_ATTRIBUTES = [:median_income, :pct_urban, :pct_black, :pct_hispanic]

# number of (quarter, dealer, model) combinations
n_delta(model::AllTrimsModel) = length(groupby(model.product_data, [:t, :f, :j]))
n_trim_params(model::AllTrimsModel) = 2length(unique(model.product_data.make))
n_demographics(_) = length(DEMOGRAPHIC_ATTRIBUTES) # median income, urban, pct_black, pct_hispanic
n_demo_product_params(x) = n_demographics(x) * n_product_attributes(x) # median income, urban, pct_black, pct_hispanic
n_product_attributes(_) = length(PRODUCT_ATTRIBUTES) # suv, truck, electric
n_demo_price_params(x) = n_demographics(x) # median income, urban
n_random_coefs(::AllTrimsModel{RandomCoefType}) = 1 # price
n_random_coefs(::AllTrimsModel{FixedCoefType}) = 0
n_dist_params(_) = 2 # distance disutility
n_mean_product_params(x) = n_product_attributes(x)
n_mean_price_params(_) = 1 # price
n_rho(_) = 1

const ORDER = [
    n_delta,
    n_trim_params,
    n_demo_product_params,
    n_demo_price_params,
    n_random_coefs,
    n_dist_params,
    n_mean_product_params,
    n_mean_price_params,
    n_rho,
]

macro pull(name, params, model)
    fname = Symbol("n_$(string(name))")
    quote
        i = findfirst(==( $fname ), ORDER)
        n_before = mapreduce(+, first(ORDER, i-1), init = 0) do f
            f($model)
        end
        @view $params[n_before .+ (1:$fname($model))]
    end
end

function δ(params::AbstractVector, model::AllTrimsModel)
    deltas = @pull delta params model # only j * t * f (lacking trims)

    # fill first trim deltas (trim = 0)
    for (δ, group) in zip(deltas, groupby(model.product_data, [:t, :f, :j]))
        group.δ .= δ
    end

    # adjust by trim parameters (trim ≥ 0)
    θs = @pull trim_params params model
    for (θ, group) in zip(partition(θs, 2), groupby(model.product_data, [:make]))
        group.δ .*= (1 .+ quantile.(Gamma(θ...), group.k))
    end

    return model.product_data.δ # J-vector
end

# demographic/product attribute interaction, demographic/price and random coeff interaction, distance disutility
function μ(params, model, data)
    zip = only(unique(data.ownrzpcd)) # scalar
    
    x_j = Matrix(model.product_data[!, PRODUCT_ATTRIBUTES]) # J x n_product_attributes
    D_i = Vector(pluck(model.agent_data, :zip_code => ByRow(==(zip)))) # n_demo-vector
    p_j = model.product_data.p # J-vector
    dist_f = model.distances[zip, :] # 1 x F
    dist_j = [ dist_f[uid] for uid in model.product_data.f ]' # 1 x J

    Πᵦ = @pull demo_product_params params model # (n_demo x n_product)
    Πₚ = @pull demo_price_params params model # (n_demo)
    σ = @pull random_coefs params model # (n_random)
    σ = isempty(σ) ? 0.0 : only(σ) # scalar
    η = @pull dist_params params model # (2)

    return D_i * (Πᵦ * x_j') .+ D_i * (Πₚ * p_j') .+ (model.u.draws * σ) .* p_j' .- exp.(η[1] .+ η[2] * dist_j) # (S x J)
end

# loglikelihood for all models, trims, quarters, dealers
# data is for one particular zip code
function loglikelihood(params::AbstractVector, model::AbstractDemandModel, data)
    δ̂ = δ(params, model)' # include mean utilities and adjustment for trim (1 x J)
    μ̂ = μ(params, model, data) # calculate individual-specific utility (S x J)
    model.u.grid[:, 1:size(δ̂, 2)] .= δ̂ .+ μ̂ # total utility (S x J), last column is outside good of zeros
    model.u.grid[:, end] .= 0.0 # outside good utility
    softmax!(model.u, dims = 2) # choice probabilities (S x J)
    p = log.(mean(model.u, dims = 1)) # log average choice probabilities (1 x J)
    return only(p'choices(model, data)) # dot product with shares * market size
end
