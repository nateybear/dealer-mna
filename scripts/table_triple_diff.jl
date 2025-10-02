using DealerMnA
using StatsBase: mean, median

using LinearAlgebra
using LaTeXTabulars
using LaTeXStrings

!isempty(ARGS) && snakemake!(ARGS[1])

function fit_plus_std(x, y; digits = 3)
    let significant = abs(x) > 1.96y, x = round(x; digits), y = round(y; digits)
        LaTeXString(
            "\\makecell{$x$(significant ? "\\textsuperscript{***}" : "") \\\\ ($y)\\\\[0.25em]}"
        )
    end
end

with_logging() do
    df = DataFrame(Arrow.Table(snakemake.input["treatment_effects"]))
    extras = JLD2.load_object(snakemake.input["extras"])
    cut = snakemake.wildcards["cut"]
    cols = [:mean_price, :total_units, :trim_rank, :mean_baseprice, :median_markup]
    transaction_uids = read_csv(snakemake.input["transaction_uids"])
    dealer_demographics = DataFrame(Arrow.Table(snakemake.input["dealer_demographics"]))
    transform!(
        transaction_uids, names(transaction_uids) .=> ByRow(UUID), renamecols = false
    )
    transaction_uids = innerjoin(
        transaction_uids, dealer_demographics, on = :dealer_uid => :uid
    )
    transaction_uids.cut = if cut == "urban"
        transaction_uids.share_urban .> median(dealer_demographics.share_urban)
    elseif cut == "exiting_seller"
        map(transaction_uids.transaction_uid) do uid
            countlines("data/affiliated_dealerships/seller_group_$(uid).txt") == 0
        end
    else
        error("Unknown cut: $cut")
    end
    table = DataFrame(
        map(cols) do col
            extra = pluck(extras, :stat => ByRow(==(col)))
            transform!(
                extra.fes,
                :reg_uid => ByRow(UUID ∘ last ∘ (@$ split(_, "_"))) => :dealer_uid,
            )
            fes = innerjoin(
                extra.fes,
                select(transaction_uids, [:dealer_uid, :cut]),
                on = :dealer_uid,
            )
            offset = @> fes begin
                groupby(:cut)
                combine([:fe_treated, :fe_synthetic] .=> mean, renamecols = false)
                transform([:fe_treated, :fe_synthetic] => ByRow(-) => :difference)
                select(:cut, :difference)
            end
            cut_true = only(offset.difference[offset.cut])
            cut_false = only(offset.difference[.!offset.cut])

            const_cols = df.interaction .== "1"
            before_cols = df.T .== -3
            after_cols = df.T .== 2

            cut_true_before = mean(df[before_cols, col]) + cut_true
            cut_true_after = mean(df[after_cols, col]) + cut_true
            cut_false_before = mean(df[const_cols .& before_cols, col]) + cut_false
            cut_false_after = mean(df[const_cols .& after_cols, col]) + cut_false

            se(idx) = sqrt(idx' * extra.V * idx)

            cut_true_before_se = se(before_cols)
            cut_true_after_se = se(after_cols)
            cut_false_before_se = se(const_cols .& before_cols)
            cut_false_after_se = se(const_cols .& after_cols)

            cut_true_diff = cut_true_after - cut_true_before
            cut_false_diff = cut_false_after - cut_false_before
            difference = cut_true_diff - cut_false_diff
            cut_true_diff_se = se(after_cols - before_cols)
            cut_false_diff_se = se((const_cols .& after_cols) - (const_cols .& before_cols))
            difference_se = se((.!const_cols .& after_cols) - (.!const_cols .& before_cols))

            return (;
                col,
                before_true = fit_plus_std(cut_true_before, cut_true_before_se),
                after_true = fit_plus_std(cut_true_after, cut_true_after_se),
                difference_true = fit_plus_std(cut_true_diff, cut_true_diff_se),
                before_false = fit_plus_std(cut_false_before, cut_false_before_se),
                after_false = fit_plus_std(cut_false_after, cut_false_after_se),
                difference_false = fit_plus_std(cut_false_diff, cut_false_diff_se),
                difference = fit_plus_std(difference, difference_se),
            )
        end,
    )

    titleize(col) = replace(string(col), "_" => " ") |> titlecase
    table.col = titleize.(table.col)

    cut = titleize(cut)

    open(snakemake.output[1], "w") do io
        latex_tabular(
            io,
            Tabular("l|" * "c"^3),
            [
                Rule(:top),
                ["", "Before", "After", "Difference"],
                collect(
                    Iterators.flatten(
                        map(eachrow(table)) do row
                            [
                                Rule(:mid),
                                [row.col],
                                Rule(:mid),
                                [
                                    cut,
                                    row.before_true,
                                    row.after_true,
                                    row.difference_true,
                                ],
                                [
                                    "Non $cut",
                                    row.before_false,
                                    row.after_false,
                                    row.difference_false,
                                ],
                                [
                                    "Difference",
                                    "",
                                    "",
                                    row.difference,
                                ],
                            ]
                        end,
                    ),
                )...,
                Rule(:mid),
                # give count of different cuts
                [
                    "N $cut",
                    string(sum(transaction_uids.cut)),
                ],
                [
                    "N Non $cut",
                    string(sum(.!transaction_uids.cut)),
                ],
                [
                    "Total N",
                    string(nrow(transaction_uids)),
                ],
                Rule(:bottom),
            ],
        )
    end
end
