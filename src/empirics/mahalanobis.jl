using StatsBase: cov

const mahalanobis_cols = [:total_population, :median_income, :share_urban, :share_black, :share_hispanic]

make_distance_fn(dealer_demographics) =
    let X = Matrix(select(dealer_demographics, mahalanobis_cols))
        Σ = cov(X)
        (d1, d2) -> let
            d1 = Vector(d1[mahalanobis_cols])
            d2 = Vector(d2[mahalanobis_cols])
            x = d1 - d2
            sqrt(x' * (Σ \ x))
        end
    end
