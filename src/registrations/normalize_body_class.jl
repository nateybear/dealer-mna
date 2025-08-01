function _normalize_body_class(raw_body_class)
    out = @> raw_body_class split("/") first
    if contains(out, r"SUV")
        "SUV/Crossover"
    elseif contains(out, r"van"i)
        "Van"
    elseif startswith(out, "Crossover") || startswith(out, "Hatchback")
        "SUV/Crossover"
    elseif out in ("Sedan", "Coupe", "Pickup")
        out
    else
        missing # all others (incomplete* variants) are rubbish
    end
end

function normalize_body_class(df::DataFrame)
    normalization = (convert$Union{String, Missing}) ∘ passmissing(_normalize_body_class)
    return transform(
        df,
        :body_class => ByRow(normalization),
        renamecols = false,
    )
end
