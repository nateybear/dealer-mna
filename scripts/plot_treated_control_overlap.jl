using DealerMnA
using Plots

!isempty(ARGS) && snakemake!(ARGS[1])

with_logging() do
    treated = read_arrow(snakemake.input["treated_registrations"])
    control = read_arrow(snakemake.input["control_registrations"])
    demographics = read_arrow(snakemake.input["dealer_demographics"])
    treated_demographics = semijoin(demographics, treated, on = :uid => :dealer_uid)
    control_demographics = semijoin(demographics, control, on = :uid => :dealer_uid)

    col = snakemake.wildcards["demographic"]

    density(
        treated_demographics[!, col],
        label = "Treated",
        lw = 2,
        c = :blue,
        alpha = 0.5,
        normalize = true,
        size = (800, 600),
    )

    density!(
        control_demographics[!, col],
        label = "Control",
        lw = 2,
        c = :red,
        alpha = 0.5,
        normalize = true,
    )

    col_pretty = replace(string(col), "_" => " ") |> titlecase

    xlabel!(col_pretty)
    ylabel!("Density")

    png(snakemake.output[1])
end
