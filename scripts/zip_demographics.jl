using DealerMnA

import Lazy: seq, partition

!isempty(ARGS) && snakemake!(ARGS[1])

function overburdened_pct(cols...)
    cols = partition(2, seq(cols))
    if all(==(0), first.(cols))
        missing
    else
        sum(last.(cols)) / sum(first.(cols))
    end
end

dropinf(x) =
    if isinf(x)
        missing
    else
        x
    end

const ACS_VARIABLES = Dict(
    :S1901 => Dict( # remove non-numeric values in zip codes with zero people ("2500-" as a median earnings...?)
        :transform => [
            :S1901_C01_012E =>
                ByRow(passmissing(x -> parse(Int64, replace(x, r"[^\d]" => "")))) =>
                    :median_earnings,
        ],
        :select => [:zip_code, :median_earnings],
    ),
    # :S2503 => Dict( # has overburdened broken out by income bracket, need to aggregate
    #     :transform => [
    #         r"^S2503_C01_0(25|28|29|32|33|36|37|40|41|44)E$" =>
    #             ByRow(overburdened_pct) => :overburdened_pct,
    #     ],
    #     :select => [:zip_code, :overburdened_pct],
    # ),
    # :S0801 => Dict(
    #     :rename => function (col)
    #         m = match(r"S0801_C01_0(02|09|10|11|12|13)E", col)
    #         isnothing(m) && return col
    #         return Dict(
    #             "02" => :drives_pct,
    #             "09" => :publictranspo_pct,
    #             "10" => :walks_pct,
    #             "11" => :bikes_pct,
    #             "12" => :taxis_pct,
    #             "13" => :wfh_pct,
    #         )[m.captures[1]]
    #     end,
    #     :select => [
    #         :zip_code,
    #         :drives_pct,
    #         :publictranspo_pct,
    #         :walks_pct,
    #         :bikes_pct,
    #         :taxis_pct,
    #         :wfh_pct,
    #     ],
    # ),
)

const CENSUS_VARIABLES = Dict(
    :P1 => Dict(
        :select => [:zip_code, :P1_001N => :population]
    ),
    :P2 => Dict(
        :transform => [
            [:P2_001N, :P2_002N] => ByRow(dropinf ∘ Float64 ∘ flip(/)) => :urban_pct
        ],
        :select => [:zip_code, :urban_pct],
    ),
    :P10 => Dict(
        :transform => [
            [:P10_002N, :P10_003N] => ByRow(dropinf ∘ Float64 ∘ flip(/)) => :white_pct,
            [:P10_002N, :P10_004N] => ByRow(dropinf ∘ Float64 ∘ flip(/)) => :black_pct,
        ],
        :select => [:zip_code, :white_pct, :black_pct],
    ),
    :P11 => Dict(
        :transform => [
            [:P11_001N, :P11_002N] =>
                ByRow(dropinf ∘ Float64 ∘ flip(/)) => :hispanic_pct,
        ],
        :select => [:zip_code, :hispanic_pct],
    ),
)

const CENSUS_OLD_VARIABLES = Dict(
    :P1 => Dict(
        :select => [:zip_code, :P001001 => :population_2010]
    ),
)

fix_zip_code!(df) = transform!(df, :NAME => (x -> last.(x, 5)) => :zip_code)

function load_demographic_file(file, config)
    df = CSV.read(file, DataFrame, skipto = 3, missingstring = ["", "-"])
    fix_zip_code!(df)
    if haskey(config, :transform)
        transform!(df, config[:transform]...)
    end
    if haskey(config, :rename)
        if config[:rename] isa Function
            DataFrames.rename!(config[:rename], df)
        else
            DataFrames.rename!(df, config[:rename]...)
        end
    end
    if haskey(config, :select)
        select!(df, config[:select]...)
    end
    return df
end

pct_change(old, new) = (new - old) / old

with_logging() do
    cd("data/Demographics") do
        for zip in readdir("raw") |> filter(endswith(".zip"))
            run(`unzip -o raw/$zip "*Data*" -d csv`) # unzip to csv directory
        end
    end
    @info "Loading demographics"
    zip_demographics = cd("data/Demographics/csv") do
        acs_files = @> string.("ACSST5Y2020.", keys(ACS_VARIABLES), "-Data.csv") zip(
            values(ACS_VARIABLES)
        )
        census_files = @> string.("DECENNIALDHC2020.", keys(CENSUS_VARIABLES), "-Data.csv") zip(
            values(CENSUS_VARIABLES)
        )
        census_old = @> string.("DECENNIALSF12010.", keys(CENSUS_OLD_VARIABLES), "-Data.csv") zip(
            values(CENSUS_OLD_VARIABLES)
        )
        return mapreduce(
            splat(load_demographic_file),
            innerjoin$(; on = :zip_code),
            union(acs_files, census_files, census_old),
        )
    end

    transform!(zip_demographics, [:population_2010, :population] => ByRow(pct_change) => :population_pct_change)

    CSV.write(snakemake.output[1], zip_demographics)
end
