function normalize_body_class(raw_body_class)
    out = @> raw_body_class split("/") first
    if contains(out, r"SUV")
        "SUV/Crossover"
    elseif contains(out, r"van"i)
        "Van"
    elseif startswith(out, "Crossover") || startswith(out, "Hatchback")
        "SUV/Crossover"
    else
        out
    end
end
