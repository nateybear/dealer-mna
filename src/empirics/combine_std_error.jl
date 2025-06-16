using StatsBase: std

# combine as if independent random variables (var(mean) = var(sum) / n^2)
combine_std_error(s) = sqrt(sum(s .^ 2) / length(s)^2)

combine_V(vs) = sum(vs) / length(vs)^2
