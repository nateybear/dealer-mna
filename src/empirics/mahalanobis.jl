using StatsBase: cov

const mahalanobis_cols = [
    :total_population,
    :median_income,
    :share_urban,
    :share_black,
    :share_hispanic,
]

function make_distance_fn(dealer_demographics)
    X = Matrix(select(dealer_demographics, mahalanobis_cols))
    Σ = cov(X)
    return (d1, d2) -> begin
        d1 = Vector(d1[mahalanobis_cols])
        d2 = Vector(d2[mahalanobis_cols])
        x = d1 - d2
        sqrt(x' * (Σ \ x))
    end
end
