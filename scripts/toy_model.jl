using DealerMnA

using LinearAlgebra
using PartialFunctions
using Distributions
using Optim
import Base.Iterators: product
import Base.Threads: @threads
import NonlinearSolve as NLS
import ForwardDiff as FD
using StaticArrays
using QuadGK

using Plots
using LaTeXStrings

########### PARAMETERS AND SETUP

F = cdf$(truncated(Exponential(1), upper = 10))
f = pdf$(truncated(Exponential(1), upper = 10))

function CS(pL, pH; vL, vH)
    ΔH = (pH - pL) / (vH - vL)
    ΔL = pL / vL
    low_surplus = quadgk(ΔL, ΔH) do θ
        (vL - pL / θ) * f(θ)
    end[1]
    high_surplus = quadgk(ΔH, 10) do θ
        (vH - pH / θ) * f(θ)
    end[1]
    return low_surplus + high_surplus
end

softplus(x) = log(1 + exp(x))
invsoftplus(y) = log(exp(y) - 1)
function to_prices(x)
    base, Δ = softplus.(x[1:2])
    return SVector{2}(base, base + Δ)
end
function from_prices(p)
    base = p[1]
    Δ = p[2] - p[1]
    return invsoftplus.(SVector{2}(base, Δ))
end
∂to_prices(x) = [1 0; 1 1] .* logit.(x[1:2])
logit(x) = exp(x) / (1 + exp(x))

# Smooth approximation of max(a, b): τ·log(exp(a/τ) + exp(b/τ))
# Prevents negative q_L when ΔH < ΔL without introducing a kink
const τ_SMOOTH = 0.1
smoothmax(a, b) =
    let m = max(a, b)
        m + τ_SMOOTH * log(exp((a - m) / τ_SMOOTH) + exp((b - m) / τ_SMOOTH))
    end
smoothmax_σ(a, b) = 1 / (1 + exp((b - a) / τ_SMOOTH)) # ∂smoothmax/∂a

function D(pL, pH; vL, vH)
    ΔH = (pH - pL) / (vH - vL)
    ΔL = pL / vL
    Δ̃H = smoothmax(ΔH, ΔL)
    return [F(Δ̃H) - F(ΔL), 1 - F(Δ̃H)]
end

function ∂D(pL, pH; vL, vH)
    ΔH = (pH - pL) / (vH - vL)
    ΔL = pL / vL
    Δ̃H = smoothmax(ΔH, ΔL)
    σ = smoothmax_σ(ΔH, ΔL)
    dΔH_dpL = -1 / (vH - vL)
    dΔH_dpH = 1 / (vH - vL)
    dΔL_dpL = 1 / vL
    dΔ̃H_dpL = σ * dΔH_dpL + (1 - σ) * dΔL_dpL
    dΔ̃H_dpH = σ * dΔH_dpH
    dD_dpL = [f(Δ̃H) * dΔ̃H_dpL - f(ΔL) * dΔL_dpL, -f(Δ̃H) * dΔ̃H_dpL]
    dD_dpH = [f(Δ̃H) * dΔ̃H_dpH, -f(Δ̃H) * dΔ̃H_dpH]
    return hcat(dD_dpL, dD_dpH)
end

function ∂D(pL; vL)
    ΔL = pL / vL
    dΔL_dpL = 1 / vL
    return -f(ΔL) * dΔL_dpL
end

function D(pL; vL)
    ΔL = pL / vL
    return 1 - F(ΔL)
end

struct QuadraticCost{T}
    kL::T
    kH::T
end
(c::QuadraticCost)(q) = 0.5 * c.kL * q[1]^2 + 0.5 * c.kH * q[2]^2

struct LinearCost{T <: AbstractVector}
    k::T
end
LinearCost(k...) = LinearCost(collect(k))
LinearCost(k::Real) = LinearCost([k])
(c::LinearCost)(q) = q ⋅ c.k

∂C(c::QuadraticCost, q) = [c.kL * q[1], c.kH * q[2]]

∂C(c::LinearCost, q) = c.k

containerType(::LinearCost{T}) where {T} = T
containerType(::QuadraticCost{T}) where {T} = Vector{T}

############### FOCS AND SOLVERS

function _SimpleRetailerFOC(dρ, ρ, params)
    (; vL, vH, C) = params
    p = to_prices(ρ)
    q = D(p[1], p[2]; vL, vH)
    return dρ .= q + ∂D(p[1], p[2]; vL, vH)' * (p - ∂C(C, q))
end
function SimpleRetailer(; C, vL, vH)
    T = containerType(C)
    init = MVector{2}(convert(T, [invsoftplus(vL / 2), invsoftplus(vH - vL)]))
    prob = NLS.NonlinearProblem(_SimpleRetailerFOC, init, (; C, vL, vH))
    return NLS.solve(prob)
end

function _SimpleRetailerFOCSingleProduct(dρ, ρ, params)
    (; vL, C) = params
    p = SVector{1}(softplus(ρ[1]))
    q = D(p[1]; vL)
    return dρ .= q + first(∂D(p[1]; vL) * (p - ∂C(C, q)))
end

function SimpleRetailerSingleProduct(; C, vL)
    T = containerType(C)
    init = MVector{1}(convert(T, [invsoftplus(vL)]))
    prob = NLS.NonlinearProblem(_SimpleRetailerFOCSingleProduct, init, (; C, vL))
    return NLS.solve(prob)
end

function _SimpleManufacturerFOC(dθ, θ, params)
    (; vL, vH, C) = params
    w = to_prices(θ)
    rparams = (; C = LinearCost(w), vL, vH)
    rsol = SimpleRetailer(; rparams...)
    drdρ = FD.jacobian(rsol.resid, rsol.u) do dρ, ρ
        _SimpleRetailerFOC(dρ, ρ, rparams)
    end
    drdθ = FD.jacobian(rsol.resid, θ) do dρ, new_θ
        new_w = to_prices(new_θ)
        new_params = merge(rparams, (; C = LinearCost(new_w)))
        _SimpleRetailerFOC(dρ, rsol.u, new_params)
    end
    dρdθ = -drdρ \ drdθ
    ρ = rsol.u
    p = to_prices(ρ)
    q = D(p[1], p[2]; vL, vH)
    dpdw = ∂to_prices(ρ) * dρdθ / ∂to_prices(θ)
    dqdw = ∂D(p[1], p[2]; vL, vH) * dpdw
    return dθ .= q + dqdw' * (w - ∂C(C, q))
end

function SimpleManufacturer(; C, vL, vH)
    p_vi = to_prices(SimpleRetailer(; C, vL, vH))
    init = MVector{2}(invsoftplus(p_vi[1] / 2), invsoftplus((p_vi[2] - p_vi[1]) / 2))
    prob = NLS.NonlinearProblem(_SimpleManufacturerFOC, init, (; C, vL, vH))
    return NLS.solve(prob)
end

function _ConstrainedRetailerFOC(dρ, ρ, params)
    (; vL, vH, C, q_bar) = params
    p = to_prices(ρ)
    λ = softplus(ρ[3])
    ∂D∂p = ∂D(p[1], p[2]; vL, vH)'
    q = D(p[1], p[2]; vL, vH)
    dρ[1:2] .= q + ∂D∂p * (p - ∂C(C, q)) + λ * ∂D∂p[:, 1]
    dρ[3] = λ * (q[1] - q_bar)
    return dρ
end

function ConstrainedRetailer(; C, vL, vH, q_bar)
    T = eltype(containerType(C))
    init = MVector{3}(T.([invsoftplus(vL / 2), invsoftplus(vH - vL), 0]))
    prob = NLS.NonlinearProblem(_ConstrainedRetailerFOC, init, (; C, vL, vH, q_bar))
    return NLS.solve(prob)
end

function DefectingRetailer(θ, vL)
    wL = softplus(θ[1])
    sol = SimpleRetailerSingleProduct(; C = LinearCost(wL), vL)
    p = softplus(sol.u[1])
    q = D(p; vL)
    Π = (p - wL) * q
    drdρ = FD.jacobian(sol.resid, sol.u) do dρ, ρ
        _SimpleRetailerFOCSingleProduct(dρ, ρ, (; C = LinearCost(wL), vL))
    end
    drdθ = FD.jacobian(sol.resid, θ[1:1]) do dρ, new_wL
        new_params = (; C = LinearCost(softplus.(new_wL)), vL)
        _SimpleRetailerFOCSingleProduct(dρ, sol.u, new_params)
    end
    dρdθ = only(-drdρ \ drdθ)
    dpdw = logit(sol.u[1]) * dρdθ / logit(θ[1])
    dqdw = ∂D(p; vL) * dpdw
    dwL = q * (dpdw - 1) + dqdw * (p - wL)
    return (Π, dwL)
end

function _ConstrainedManufacturer(dθ, θ, params)
    (; vL, vH, C, q_bar) = params
    w = to_prices(θ)
    μ = softplus(θ[3])
    rparams = (; C = LinearCost(w), vL, vH, q_bar)
    rsol = ConstrainedRetailer(; rparams...)
    drdρ = FD.jacobian(rsol.resid, rsol.u) do dρ, ρ
        _ConstrainedRetailerFOC(dρ, ρ, rparams)
    end
    drdθ = FD.jacobian(rsol.resid, θ[1:2]) do dρ, new_θ
        new_w = to_prices(new_θ)
        new_params = merge(rparams, (; C = LinearCost(new_w)))
        _ConstrainedRetailerFOC(dρ, rsol.u, new_params)
    end
    dρdθ = -drdρ \ drdθ
    p = to_prices(rsol.u)
    q = D(p[1], p[2]; vL, vH)
    Jρ = [∂to_prices(rsol.u) zeros(2); 0 0 logit(rsol.u[3])]
    dpdθ = Jρ * dρdθ                                          # 3×2, ∂(p₁,p₂,λ)/∂θ
    dqdθ = ∂D(p[1], p[2]; vL, vH) * dpdθ[1:2, :]             # 2×2, ∂q/∂θ
    Jw = ∂to_prices(θ)                                        # 2×2, ∂w/∂θ
    Π_comply = (p - w) ⋅ q
    dcomplydθ = dqdθ' * (p - w) + (dpdθ[1:2, :] - Jw)' * q
    Π_defect, ddefectdwL = DefectingRetailer(θ, vL)
    dθ[1:2] .=
        Jw' * q + dqdθ' * (w - ∂C(C, q)) + μ * dcomplydθ -
        μ * vcat(ddefectdwL * logit(θ[1]), 0)
    dθ[3] = μ * (Π_comply - Π_defect)
    return dθ
end

function ConstrainedManufacturer(; C, vL, vH, q_bar)
    init = MVector{3}(Float64[invsoftplus(vL), invsoftplus(vH - vL), -1.0])
    prob = NLS.NonlinearProblem(_ConstrainedManufacturer, init, (; C, vL, vH, q_bar))
    return NLS.solve(prob)
end

function simple_manufacturer(; C, vL, vH)
    w = to_prices(SimpleManufacturer(; C, vL, vH))
    p = to_prices(SimpleRetailer(; C = LinearCost(w), vL, vH))
    q = D(p[1], p[2]; vL, vH)
    surplus = q ⋅ p - C(q)
    Π_manufacturer = w ⋅ q - C(q)
    Π_retailer = surplus - Π_manufacturer
    return (; p, q, w, surplus, Π_manufacturer, Π_retailer, CS = CS(p[1], p[2]; vL, vH))
end

################# SUMMARY FUNCTIONS

function vertically_integrated(; C, vL, vH)
    p = to_prices(SimpleRetailer(; C, vL, vH))
    q = D(p[1], p[2]; vL, vH)
    surplus = p ⋅ q - C(q)
    return (; p, q, surplus, CS = CS(p[1], p[2]; vL, vH))
end

function allocating_manufacturer(; C, vL, vH)
    q_search = optimize(0, 0.5) do q
        try
            w = to_prices(ConstrainedManufacturer(; C, vL, vH, q_bar = q))
            p = to_prices(ConstrainedRetailer(; C = LinearCost(w), vL, vH, q_bar = q))
            q = D(p[1], p[2]; vL, vH)
            return -(w ⋅ q - C(q))
        catch e
            @debug "Optimization error" q e
            return Inf
        end
    end
    q_bar = Optim.minimizer(q_search)
    w = to_prices(ConstrainedManufacturer(; C, vL, vH, q_bar))
    p = to_prices(ConstrainedRetailer(; C = LinearCost(w), vL, vH, q_bar))
    q = D(p...; vL, vH)
    surplus = p ⋅ q - C(q)
    p_defection = softplus(SimpleRetailerSingleProduct(; C = LinearCost(w[1]), vL).u[1])
    q_defection = D(p_defection; vL)
    Π_defect = (p_defection - w[1]) * q_defection
    Π_comply = (p - w) ⋅ q
    Π_manufacturer = w ⋅ q - C(q)
    return (; w, p, q, surplus, q_bar, Π_comply, Π_defect, Π_manufacturer, CS = CS(p[1], p[2]; vL, vH))
end

recovered_surplus(d::Dict) = recovered_surplus(d["c_L"], d["c_H"], d["v_L"], d["v_H"])

function recovered_surplus(cL, cH, vL, vH)
    C = QuadraticCost(cL, cH)
    vi = vertically_integrated(; C, vL, vH)
    dm = simple_manufacturer(; C, vL, vH)
    alloc = allocating_manufacturer(; C, vL, vH)
    return (; vi, dm, alloc, recovered = (alloc.surplus - dm.surplus) / (vi.surplus - dm.surplus))
end

################ MAIN
function (@main)(_)
    snakemake!()

    RESOLUTION = 50

    with_logging() do
        data = Dict(
            "c_L" => 2.0,
            "c_H" => 4.0,
            "v_L" => 2.0,
            "v_H" => 3.0,
        )

        makerange(label) = range(data[label] - 0.1, data[label] + 0.1, length = RESOLUTION)
        JLD2.save_object(snakemake.output["jld2"], recovered_surplus(data))
        grids = Dict{Tuple{String, String}, Matrix{Float64}}()
        # for (label1, label2) in [("c_L", "c_H"), ("v_L", "v_H"), ("c_L", "v_L"), ("c_H", "v_H")]
        #     range1 = makerange(label1)
        #     range2 = makerange(label2)
        #     out = zeros(RESOLUTION, RESOLUTION)
        #     @info "Outer loop" label1 label2
        #     value_iterator = collect(product(enumerate(range1), enumerate(range2)))
        #     for ((i1, value1), (i2, value2)) in value_iterator
        #         @info "Inner loop $(label1) = $value1, $(label2) = $value2"
        #         d = copy(data)
        #         d[label1] = value1
        #         d[label2] = value2
        #         out[i1, i2] = recovered_surplus(d).recovered
        #     end
        #     grids[(label1, label2)] = out
        # end

        # max_recovered = maximum(maximum(grid) for grid in values(grids))
        # min_recovered = minimum(minimum(grid) for grid in values(grids))
        # @info "Making plots"
        # for ((label1, label2), grid) in grids
        #     range1 = makerange(label1)
        #     range2 = makerange(label2)
        #     contourf(
        #         range1,
        #         range2,
        #         grid,
        #         xlabel = LaTeXString("\$" * label1 * "\$"),
        #         ylabel = LaTeXString("\$" * label2 * "\$"),
        #         colorbar_title = "Frac. Surplus Recovered",
        #         color = :redsblues,
        #         clim = (min_recovered, max_recovered),
        #         levels = 50,
        #         size = (800, 600),
        #     )
        #     png(snakemake.output["png_$(label1)_$(label2)"])
        # end

        @info "Tracing profit plot"
        (; vi, dm, alloc) = recovered_surplus(data)
        q_grid = range(0, 0.5, length = 100)
        C = QuadraticCost(data["c_L"], data["c_H"])
        vL, vH = data["v_L"], data["v_H"]
        y = map(q_grid) do q_bar
            try
                cm = ConstrainedManufacturer(; C, vL, vH, q_bar = q_bar)
                w = to_prices(cm)
                cr = ConstrainedRetailer(; C = LinearCost(w), vL, vH, q_bar = q_bar)
                p = to_prices(cr)
                q = D(p[1], p[2]; vL, vH)
                surplus = p ⋅ q - C(q)
                return (;
                    Π_manufacturer = w ⋅ q - C(q),
                    Π_retailer = surplus - (w ⋅ q - C(q)),
                    surplus,
                    w,
                    p,
                    q,
                    λ = softplus(cr.u[3]),
                    μ = softplus(cm.u[3]),
                    q_bar = q_bar,
                    CS = CS(p[1], p[2]; vL, vH),
                )
            catch
                return missing
            end
        end

        y = DataFrame(filter(!ismissing, y))

        plot(
            y.q_bar,
            y.Π_manufacturer,
            xlabel = L"\underbar{q}_L",
            ylabel = L"\pi^M",
            label = "Profit with Sales Target",
            legend = :topright,
        )
        vline!([alloc.q_bar], label = "Optimal Sales Target")
        vline!([vi.q[1]], label = "Vertically Integrated Output")
        vline!([dm.q[1]], label = "Double Marginalization Output")
        png(snakemake.output["profit_plot"])

        plot(
            y.q_bar,
            y.μ,
            xlabel = L"\underbar{q}_L",
            ylabel = L"\mu",
            label = "Lagrange Multiplier on Retailer Incentive Constraint",
            legend = :topright,
        )
        vline!([alloc.q_bar], label = "Optimal Sales Target")
        vline!([dm.q[1]], label = "Double Marginalization Output")
        png(snakemake.output["mu_plot"])

        plot(
            y.q_bar,
            y.λ,
            xlabel = L"\underbar{q}_L",
            ylabel = L"\lambda",
            label = "Lagrange Multiplier on Sales Target Constraint",
            legend = :right,
        )
        vline!([alloc.q_bar], label = "Optimal Sales Target")
        vline!([dm.q[1]], label = "Double Marginalization Output")
        png(snakemake.output["lambda_plot"])

        plot(
            y.q_bar,
            first.(y.w),
            xlabel = L"\underbar{q}_L",
            ylabel = L"w_L",
            label = "Wholesale Price of Low-Quality Product",
            legend = :topright,
        )
        vline!([alloc.q_bar], label = "Optimal Sales Target")
        png(snakemake.output["wL_plot"])

        plot(
            y.q_bar,
            last.(y.w),
            xlabel = L"\underbar{q}_L",
            ylabel = L"w_H",
            label = "Wholesale Price of High-Quality Product",
            legend = :bottomright,
        )
        vline!([alloc.q_bar], label = "Optimal Sales Target")
        png(snakemake.output["wH_plot"])

        plot(
            y.q_bar,
            y.Π_retailer,
            xlabel = L"\underbar{q}_L",
            ylabel = L"\pi^R",
            label = "Profit of Low-Quality Product",
            legend = :bottomright,
        )
        vline!([alloc.q_bar], label = "Optimal Sales Target")
        png(snakemake.output["piR_plot"])

        plot(
            y.q_bar,
            y.surplus,
            xlabel = L"\underbar{q}_L",
            ylabel = "Total Surplus",
            label = "Total Surplus",
            legend = :bottomright,
        )
        vline!([alloc.q_bar], label = "Optimal Sales Target")
        png(snakemake.output["surplus_plot"])

        plot(
            y.q_bar,
            last.(y.q),
            xlabel = L"\underbar{q}_L",
            ylabel = L"q_H",
            label = "High-Quality Product Quantity",
            legend = :bottomright,
        )
        vline!([alloc.q_bar], label = "Optimal Sales Target")
        png(snakemake.output["qH_plot"])
    end
    return nothing
end
