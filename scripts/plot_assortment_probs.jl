using DealerMnA
using Plots
using StatsBase: cov
using Distributions
using LinearAlgebra

D = Empirics.DEFAULT_DEMOGRAPHICS

selectunique(X, cols) = select(unique(X, cols), cols)

struct Mahalanobis{T}
    Σ::T
end
mahalanobis(X) = Mahalanobis(qr(cov(Matrix(X))))

GaussMahalanobis(m::Mahalanobis, h = 0.5) = x -> pdf(Normal(), sqrt(x ⋅ (m.Σ \ x)) / h)

nadarayawatson(K, X, y, Xstar = X) =
    map(eachrow(Xstar)) do x
        x = reshape(x, 1, :)
        W = Diagonal(K.(eachrow((X .- x))))
        βₓ = (X'W * X) \ (X'W * y)
        return only(x * βₓ)
    end

function make_panel(fords, dealers)
    out = @> fords groupby([:t, :dealer_uid, :model, :model_year, :trim_rank]) combine(nrow)
    period_sales = @> out groupby([:t, :dealer_uid]) combine(:nrow => sum => :lag_sales)
    out = @> period_sales transform(:t => ByRow(t -> t + Quarter(1)) => :t) innerjoin(
        out, on = [:dealer_uid, :t]
    )
    return mapreduce(vcat, groupby(out, :t)) do sales
        J = @> sales selectunique([:model, :model_year, :trim_rank])
        J.is_new_model_year = J.model_year .== maximum(J.model_year)
        F = @> sales selectunique(:dealer_uid)
        JF = crossjoin(J, F)
        JF.t .= first(sales.t)
        leftjoin!(JF, select(unique(sales, :dealer_uid), :dealer_uid, :lag_sales), on = :dealer_uid) # attach lagged sales
        leftjoin!(
            JF,
            select(sales, Not(:lag_sales)),
            on = [:t, :dealer_uid, :model, :model_year, :trim_rank],
        )
        JF.nrow = coalesce.(JF.nrow, 0)
        JF.is_sold = JF.nrow .> 0
        leftjoin!(
            JF,
            select(dealers, :uid, D),
            on = :dealer_uid => :uid,
        )
        return JF
    end
end
