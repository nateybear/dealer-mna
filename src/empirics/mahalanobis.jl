using StatsBase: cov

const mahalanobis_cols = [
    :total_population,
    :change_population,
    :median_income,
    :share_urban,
    :share_black,
    :share_hispanic,
    :closest_first,
    :closest_second,
    :closest_third,
]

function make_distance_fn(dealer_demographics, cols = mahalanobis_cols)
    X = Matrix(select(dealer_demographics, cols))
    Σ = cov(X)
    return (d1, d2) -> begin
        d1 = Vector(d1[cols])
        d2 = Vector(d2[cols])
        x = d1 - d2
        sqrt(x' * (Σ \ x))
    end
end
