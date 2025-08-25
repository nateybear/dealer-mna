import glob
import pandas as pd
from itertools import chain

julia_src = glob.glob("src/**/*.jl")
julia_dealers = glob.glob("src/dealers/**/*.jl")
julia_empirics = glob.glob("src/empirics/**/*.jl")


wildcard_constraints:
    transaction_uid="[0-9a-f\\-]+",
    dealer_uid="[0-9a-f\\-]+",
    statistic="[a-z_]+",
    cut="[a-z_]+",
    folder="[a-z_]+",
    buy_sell="[a-z]+",
    i="[0-9]+",


rule show_python_info:
    run:
        import sys

        print(sys.executable)
        print(sys.version)


rule vins_decoded:
    input:
        "data/vins_decoded.csv",
        "scripts/vins_decoded.jl",
    output:
        "data/arrow/vins_decoded.arrow",
    log:
        "logs/vins_decoded.log",
    run:
        julia(
            "scripts/vins_decoded.jl",
            input=input,
            output=output,
            log=log,
        )


rule registrations:
    input:
        "scripts/registrations.jl",
        registration_files=glob.glob("data/Sales/*.csv"),
        vins_decoded="data/arrow/vins_decoded.arrow",
    output:
        registrations="data/arrow/registrations.arrow",
    log:
        "logs/registrations.log",
    run:
        julia(
            "scripts/registrations.jl",
            input=input,
            output=output,
            log=log,
        )


rule dealers_init:
    input:
        *julia_dealers,
        "src/dealers/Dealers.jl",
        "scripts/dealers_init.jl",
        unmatched_registrations="data/arrow/unmatched_registrations.arrow",
        popular_models="data/arrow/popular_models.arrow",
    output:
        "data/dealers_init.jld2",
    log:
        "logs/dealers_init.log",
    run:
        julia("scripts/dealers_init.jl", input=input, output=output, log=log)


# a "halfway" step that's necessary to avoid a cyclic dependency with dealers_init
# (dealers_init needs to know popular models, which depends on count of new_registrations,
# but new_registrations needs to match against dealers. Count popular_models from
# not-necessary-matched new registrations)
rule unmatched_registrations:
    input:
        "scripts/unmatched_registrations.jl",
        registrations="data/arrow/registrations.arrow",
    output:
        unmatched_registrations="data/arrow/unmatched_registrations.arrow",
        popular_models="data/arrow/popular_models.arrow",
    log:
        "logs/unmatched_registrations.log",
    run:
        julia(
            "scripts/unmatched_registrations.jl",
            input=input,
            output=output,
            log=log,
        )


rule new_registrations:
    input:
        "scripts/new_registrations.jl",
        unmatched_registrations="data/arrow/unmatched_registrations.arrow",
        dealers_init="data/dealers_init.jld2",
    output:
        new_registrations="data/arrow/new_registrations.arrow",
    log:
        "logs/new_registrations.log",
    run:
        julia(
            "scripts/new_registrations.jl",
            input=input,
            output=output,
            log=log,
        )


rule price_table:
    input:
        "scripts/price_table.jl",
        "src/registrations/normalize_price.jl",
        new_registrations="data/arrow/new_registrations.arrow",
        popular_models="data/arrow/popular_models.arrow",
    output:
        "data/price_table.arrow",
    log:
        "logs/price_table.log",
    run:
        julia("scripts/price_table.jl", input=input, output=output, log=log)


checkpoint all_manufacturers:
    input:
        "scripts/all_manufacturers.jl",
        price_table="data/price_table.arrow",
    output:
        "data/all_manufacturers.txt",
    log:
        "logs/all_manufacturers.log",
    run:
        julia("scripts/all_manufacturers.jl", input=input, output=output, log=log)


def get_all_manufacturers(wildcards):
    # Get the output file from the rule at runtime
    with open(checkpoints.all_manufacturers.get().output[0], "r") as f:
        return f.read().splitlines()


rule transactions_for_analysis:
    input:
        *julia_dealers,
        "src/empirics/Empirics.jl",
        "scripts/transactions_for_analysis.jl",
        dealers_init="data/dealers_init.jld2",
        price_table="data/price_table.arrow",
        new_registrations="data/arrow/new_registrations.arrow",
        popular_models="data/arrow/popular_models.arrow",
    output:
        "data/transactions_for_analysis.jld2",
    log:
        "logs/transactions_for_analysis.log",
    run:
        julia(
            "scripts/transactions_for_analysis.jl", input=input, output=output, log=log
        )


checkpoint transaction_uids:
    input:
        "scripts/transaction_uids.jl",
        "data/statistics.txt",
        transactions_for_analysis="data/transactions_for_analysis.jld2",
        new_registrations="data/arrow/new_registrations.arrow",
        popular_models="data/arrow/popular_models.arrow",
        dealers_init="data/dealers_init.jld2",
    output:
        "data/transaction_uids.csv",
    log:
        "logs/transaction_uids.log",
    run:
        julia("scripts/transaction_uids.jl", input=input, output=output, log=log)


checkpoint statistics:
    input:
        "scripts/statistics.jl",
        "src/empirics/event_study.jl",
    output:
        "data/statistics.txt",
    log:
        "logs/statistics.log",
    run:
        julia("scripts/statistics.jl", output=output, log=log)


rule treated_registrations:
    input:
        "src/empirics/registrations.jl",
        "src/empirics/Empirics.jl",
        "scripts/treated_registrations.jl",
        transactions_for_analysis="data/transactions_for_analysis.jld2",
        dealers_init="data/dealers_init.jld2",
        price_table="data/price_table.arrow",
        new_registrations="data/arrow/new_registrations.arrow",
        popular_models="data/arrow/popular_models.arrow",
    output:
        before="data/treated/{folder}/last_registrations_before_{transaction_uid}_{dealer_uid}.arrow",
        after="data/treated/{folder}/first_registrations_after_{transaction_uid}_{dealer_uid}.arrow",
    log:
        "logs/{folder}_{transaction_uid}_{dealer_uid}.log",
    run:
        julia(
            "scripts/treated_registrations.jl",
            input=input,
            output=output,
            log=log,
            wildcards=wildcards,
        )


rule control_registrations:
    input:
        "src/empirics/registrations.jl",
        "src/empirics/Empirics.jl",
        "scripts/control_registrations.jl",
        transactions_for_analysis="data/transactions_for_analysis.jld2",
        dealers_init="data/dealers_init.jld2",
        price_table="data/price_table.arrow",
        new_registrations="data/arrow/new_registrations.arrow",
        popular_models="data/arrow/popular_models.arrow",
    output:
        before="data/control/{folder}/last_registrations_before_{transaction_uid}_{dealer_uid}.arrow",
        after="data/control/{folder}/first_registrations_after_{transaction_uid}_{dealer_uid}.arrow",
    log:
        "logs/{folder}_{transaction_uid}_{dealer_uid}.log",
    run:
        julia(
            "scripts/control_registrations.jl",
            input=input,
            output=output,
            log=log,
            wildcards=wildcards,
        )


rule dealer_demographics:
    input:
        "scripts/dealer_demographics.jl",
        dealers_init="data/dealers_init.jld2",
    output:
        "data/dealer_demographics.arrow",
    log:
        "logs/dealer_demographics.log",
    run:
        julia("scripts/dealer_demographics.jl", input=input, output=output, log=log)


rule treatment_effects:
    input:
        "src/empirics/event_study.jl",
        "scripts/treatment_effects.jl",
        treated_before="data/treated/{folder}/last_registrations_before_{transaction_uid}_{dealer_uid}.arrow",
        treated_after="data/treated/{folder}/first_registrations_after_{transaction_uid}_{dealer_uid}.arrow",
        control_before="data/control/{folder}/last_registrations_before_{transaction_uid}_{dealer_uid}.arrow",
        control_after="data/control/{folder}/first_registrations_after_{transaction_uid}_{dealer_uid}.arrow",
        dealer_demographics="data/dealer_demographics.arrow",
        dealers_init="data/dealers_init.jld2",
        new_registrations="data/arrow/new_registrations.arrow",
        popular_models="data/arrow/popular_models.arrow",
    output:
        treatment_effects="data/treatment_effects/{folder}/{transaction_uid}/{dealer_uid}.arrow",
        extra_stats="data/treatment_effects/{folder}/{transaction_uid}/{dealer_uid}_extras.jld2",
    log:
        "logs/treatment_effects_{folder}_{transaction_uid}_{dealer_uid}.log",
    run:
        julia(
            "scripts/treatment_effects.jl",
            input=input,
            output=output,
            log=log,
            wildcards=wildcards,
        )


# these generate sets of uids for each transaction
checkpoint affiliated_dealerships:
    input:
        "data/transaction_uids.csv",
        "scripts/affiliated_dealerships.jl",
        transactions="data/transactions_for_analysis.jld2",
        dealers_init="data/dealers_init.jld2",
        new_registrations="data/arrow/new_registrations.arrow",
        popular_models="data/arrow/popular_models.arrow",
    output:
        buyer_group="data/affiliated_dealerships/buyer_group_{transaction_uid}.txt",
        seller_group="data/affiliated_dealerships/seller_group_{transaction_uid}.txt",
    log:
        "logs/affiliated_dealerships_{transaction_uid}.log",
    run:
        julia(
            "scripts/affiliated_dealerships.jl",
            input=input,
            output=output,
            log=log,
            wildcards=wildcards,
        )


# get three closest dealerships for each transaction
checkpoint closest_dealerships:
    input:
        "data/transaction_uids.csv",
        "scripts/closest_dealerships.jl",
        dealers_init="data/dealers_init.jld2",
        transactions="data/transactions_for_analysis.jld2",
        new_registrations="data/arrow/new_registrations.arrow",
        popular_models="data/arrow/popular_models.arrow",
    output:
        "data/closest_dealerships/{transaction_uid}_{dealer_uid}.txt",
    log:
        "logs/closest_dealerships_{transaction_uid}_{dealer_uid}.log",
    run:
        julia(
            "scripts/closest_dealerships.jl",
            input=input,
            output=output,
            log=log,
            wildcards=wildcards,
        )


def get_related_dealerships(wildcards):
    transaction_uids = pd.read_csv(checkpoints.transaction_uids.get().output[0])
    out = set()
    for t in transaction_uids.itertuples():
        out.add(f"data/closest_dealerships/{t.transaction_uid}_{t.dealer_uid}.txt")
        out.add(f"data/affiliated_dealerships/buyer_group_{t.transaction_uid}.txt")
        out.add(f"data/affiliated_dealerships/seller_group_{t.transaction_uid}.txt")

    return list(out)


# used to force the checkpoints to run
checkpoint related_dealerships:
    input:
        "data/transaction_uids.csv",
        get_related_dealerships,
        new_registrations="data/arrow/new_registrations.arrow",
        popular_models="data/arrow/popular_models.arrow",
    output:
        "data/related_dealerships.txt",
    shell:
        "touch {output}"


get_treatment_effects = lambda wildcards: sorted(
    glob.glob(f"data/treatment_effects/{wildcards.folder}/**/*.arrow")
)


get_extras = lambda wildcards: sorted(
    glob.glob(f"data/treatment_effects/{wildcards.folder}/**/*_extras.jld2")
)


def get_all_treatment_effects(wildcards):
    """
    Get treatment effects for all related dealerships (treated, affiliated, and closest dealers).
    This includes the side effect of generating affiliated and closest dealer treatment effects
    without needing to generate plots.
    """
    # Get the output files from checkpoints at runtime
    transaction_uids = pd.read_csv(checkpoints.transaction_uids.get().output[0])

    # Force the checkpoints to run to get related dealerships
    related_files = checkpoints.related_dealerships.get().output

    out = set()
    for t in transaction_uids.itertuples():
        # Add treated dealer treatment effects
        out.add(
            f"data/treatment_effects/treated_dealers/{t.transaction_uid}/{t.dealer_uid}.arrow"
        )

        # Add closest dealers treatment effects
        closest_dealers_file = checkpoints.closest_dealerships.get(
            transaction_uid=t.transaction_uid, dealer_uid=t.dealer_uid
        ).output[0]
        with open(closest_dealers_file, "r") as f:
            closest_dealers = f.read().splitlines()
        for d in closest_dealers:
            out.add(
                f"data/treatment_effects/closest_dealers/{t.transaction_uid}/{d}.arrow"
            )

        # Add buyer group treatment effects
        buyer_group_file = checkpoints.affiliated_dealerships.get(
            transaction_uid=t.transaction_uid
        ).output["buyer_group"]
        with open(buyer_group_file, "r") as f:
            buyer_group = f.read().splitlines()
        for d in buyer_group:
            out.add(f"data/treatment_effects/buyer/{t.transaction_uid}/{d}.arrow")

        # Add seller group treatment effects
        seller_group_file = checkpoints.affiliated_dealerships.get(
            transaction_uid=t.transaction_uid
        ).output["seller_group"]
        with open(seller_group_file, "r") as f:
            seller_group = f.read().splitlines()
        for d in seller_group:
            out.add(f"data/treatment_effects/seller/{t.transaction_uid}/{d}.arrow")

    return list(out)


rule treated_event_study_plot:
    input:
        "scripts/event_study_plot.jl",
        treatment_effects="data/treatment_effects/treated_dealers/{transaction_uid}/{dealer_uid}.arrow",
        extra_stats="data/treatment_effects/treated_dealers/{transaction_uid}/{dealer_uid}_extras.jld2",
    output:
        "plots/event_study/{statistic}/treated_dealers/{transaction_uid}_{dealer_uid}.png",
    log:
        "logs/event_study_plot_treated_{statistic}_{transaction_uid}_{dealer_uid}.log",
    run:
        julia(
            "scripts/event_study_plot.jl",
            input=input,
            output=output,
            log=log,
            wildcards=wildcards,
        )


rule affiliated_event_study_plot:
    input:
        "scripts/event_study_plot.jl",
        lambda w: checkpoints.affiliated_dealerships.get(
            transaction_uid=w.transaction_uid
        ).output[f"{w.buy_sell}_group"],
        treatment_effects="data/treatment_effects/{buy_sell}/{transaction_uid}/{dealer_uid}.arrow",
        extra_stats="data/treatment_effects/{buy_sell}/{transaction_uid}/{dealer_uid}_extras.jld2",
    output:
        "plots/event_study/{statistic}/{buy_sell}/{transaction_uid}_{dealer_uid}.png",
    log:
        "logs/event_study_plot_{buy_sell}_{statistic}_{transaction_uid}_{dealer_uid}.log",
    run:
        julia(
            "scripts/event_study_plot.jl",
            input=input,
            output=output,
            log=log,
            wildcards=wildcards,
        )


rule closest_dealer_event_study_plot:
    input:
        "scripts/event_study_plot.jl",
        lambda w: checkpoints.closest_dealerships.get(
            transaction_uid=w.transaction_uid, dealer_uid=w.dealer_uid
        ).output[0],
        treatment_effects="data/treatment_effects/closest_dealers/{transaction_uid}/{dealer_uid}.arrow",
        extra_stats="data/treatment_effects/closest_dealers/{transaction_uid}/{dealer_uid}_extras.jld2",
    output:
        "plots/event_study/{statistic}/closest_dealers/{transaction_uid}_{i}_{dealer_uid}.png",
    log:
        "logs/event_study_plot_closest_{statistic}_{transaction_uid}_{i}_{dealer_uid}.log",
    run:
        julia(
            "scripts/event_study_plot.jl",
            input=input,
            output=output,
            log=log,
            wildcards=wildcards,
        )


def get_event_study_plots(wildcards):
    # Get the output file from the checkpoint at runtime
    transaction_uids = pd.read_csv(checkpoints.transaction_uids.get().output[0])
    with open(checkpoints.statistics.get().output[0], "r") as f:
        statistics = f.read().splitlines()

    # force the checkpoints to run
    related_files = checkpoints.related_dealerships.get().output
    out = set()
    for t in transaction_uids.itertuples():
        for statistic in statistics:
            out.add(
                f"plots/event_study/{statistic}/treated_dealers/{t.transaction_uid}_{t.dealer_uid}.png"
            )
            closest_dealers_file = checkpoints.closest_dealerships.get(
                transaction_uid=t.transaction_uid, dealer_uid=t.dealer_uid
            ).output[0]
            with open(closest_dealers_file, "r") as f:
                closest_dealers = f.read().splitlines()
            for i, d in enumerate(closest_dealers):
                out.add(
                    f"plots/event_study/{statistic}/closest_dealers/{t.transaction_uid}_{i+1}_{d}.png"
                )

            buyer_group_file = checkpoints.affiliated_dealerships.get(
                transaction_uid=t.transaction_uid
            ).output["buyer_group"]
            with open(buyer_group_file, "r") as f:
                buyer_group = f.read().splitlines()
            for d in buyer_group:
                out.add(
                    f"plots/event_study/{statistic}/buyer/{t.transaction_uid}_{d}.png"
                )

            seller_group_file = checkpoints.affiliated_dealerships.get(
                transaction_uid=t.transaction_uid
            ).output["seller_group"]
            with open(seller_group_file, "r") as f:
                seller_group = f.read().splitlines()
            for d in seller_group:
                out.add(
                    f"plots/event_study/{statistic}/seller/{t.transaction_uid}_{d}.png"
                )

    return list(out)


rule event_study_plots:
    input:
        "data/statistics.txt",
        "data/transaction_uids.csv",
        "data/related_dealerships.txt",
        get_event_study_plots,


rule event_study_cut_plot:
    input:
        "scripts/event_study_plot.jl",
        get_event_study_plots,  # make sure all the plots are made before making cuts, since it relies on glob
        treatment_effects=get_treatment_effects,
        extras=get_extras,
        dealer_demographics="data/dealer_demographics.arrow",
    output:
        "plots/event_study/{statistic}/{folder}/cuts/{cut}.png",
    log:
        "logs/event_study_{folder}_cut_plot_{statistic}_{cut}.log",
    run:
        julia(
            "scripts/event_study_plot.jl",
            input=input,
            output=output,
            log=log,
            wildcards=wildcards,
        )


def get_event_study_cuts(wildcards):
    # Get the output file from the checkpoint at runtime
    with open(checkpoints.statistics.get().output[0], "r") as f:
        statistics = f.read().splitlines()
    return expand(
        "plots/event_study/{statistic}/{folder}/cuts/{cut}.png",
        statistic=statistics,
        cut=[
            "grand_mean",
            "share_urban",
            "share_black",
            "share_hispanic",
            "median_income",
        ],
        folder=[
            "treated_dealers",
            "closest_dealers",
            "buyer",
            "seller",
        ],
    )


rule event_study_cut_plots:
    input:
        "data/statistics.txt",
        "data/transaction_uids.csv",
        "data/related_dealerships.txt",
        get_event_study_cuts,


rule dealer_group_size_histogram:
    input:
        "scripts/dealer_group_size_histogram.jl",
        dealers_init="data/dealers_init.jld2",
        new_registrations="data/arrow/new_registrations.arrow",
        popular_models="data/arrow/popular_models.arrow",
    output:
        "plots/dealer_group_size_histogram.png",
    log:
        "logs/dealer_group_size_histogram.log",
    run:
        julia(
            "scripts/dealer_group_size_histogram.jl",
            input=input,
            output=output,
            log=log,
        )


rule table_body_make_count:
    input:
        "scripts/table_body_make_count.jl",
        new_registrations="data/arrow/new_registrations.arrow",
    output:
        "tables/body_make_count.tex",
    log:
        "logs/table_body_make_count.log",
    run:
        julia("scripts/table_body_make_count.jl", input=input, output=output, log=log)


rule table_te_crosstabs:
    input:
        "scripts/table_te_crosstabs.jl",
        treatment_effects=get_all_treatment_effects,
        transaction_uids="data/transaction_uids.csv",
    output:
        "tables/te_crosstabs.tex",
    log:
        "logs/table_te_crosstabs.log",
    run:
        julia("scripts/table_te_crosstabs.jl", input=input, output=output, log=log)


rule table_te_drilldown:
    input:
        "scripts/table_te_drilldown.jl",
        get_related_dealerships,
        treatment_effects=get_all_treatment_effects,
        transaction_uids="data/transaction_uids.csv",
    output:
        q_increased="tables/te_drilldown_q_increased.tex",
        q_decreased="tables/te_drilldown_q_decreased.tex",
        group_sizes="tables/te_drilldown_group_sizes.tex",
    log:
        "logs/table_te_drilldown.log",
    run:
        julia("scripts/table_te_drilldown.jl", input=input, output=output, log=log)


rule table_effect_on_closest:
    input:
        "scripts/table_effect_on_closest.jl",
        get_related_dealerships,
        treatment_effects=get_all_treatment_effects,
        transaction_uids="data/transaction_uids.csv",
        dealers_init="data/dealers_init.jld2",
    output:
        table_regression="tables/effect_on_closest.tex",
        scatter_sales="plots/effect_on_closest_scatter.png",
    log:
        "logs/table_effect_on_closest.log",
    run:
        julia("scripts/table_effect_on_closest.jl", input=input, output=output, log=log)


rule group_market_share_changes:
    input:
        "scripts/group_market_share_changes.jl",
        "data/statistics.txt",
        "data/transaction_uids.csv",
        "data/related_dealerships.txt",
        "src/empirics/market_share.jl",
        get_all_treatment_effects,
        dealers_init="data/dealers_init.jld2",
        transaction_uids="data/transaction_uids.csv",
        transactions_for_analysis="data/transactions_for_analysis.jld2",
        new_registrations="data/arrow/new_registrations.arrow",
    output:
        "data/group_market_share_changes.arrow",
    log:
        "logs/group_market_share_changes.log",
    run:
        julia(
            "scripts/group_market_share_changes.jl",
            input=input,
            output=output,
            log=log,
        )


rule plot_market_share_change_te:
    input:
        "scripts/plot_market_share_changes_tes.jl",
        group_market_share_changes="data/group_market_share_changes.arrow",
    output:
        "plots/market_share_change_te_{col}.png",
    log:
        "logs/plot_market_share_change_te_{col}.log",
    run:
        julia(
            "scripts/plot_market_share_changes_tes.jl",
            input=input,
            output=output,
            log=log,
            wildcards=wildcards,
        )


def get_plot_market_share_changes_tes(wildcards):
    return expand(
        "plots/market_share_change_te_{col}.png",
        col=[
            "mean_price",
            "median_markup",
            "mean_baseprice",
            "total_units",
            "trim_rank",
        ],
    )


rule plot_trim_distn_manufacturer:
    input:
        "scripts/plot_trim_distn_manufacturer.jl",
        price_table="data/price_table.arrow",
    output:
        rank="plots/trim_distn/{manufacturer}_rank.png",
        relprice="plots/trim_distn/{manufacturer}_relprice.png",
    log:
        "logs/plot_trim_distn_{manufacturer}.log",
    run:
        julia(
            "scripts/plot_trim_distn_manufacturer.jl",
            input=input,
            output=output,
            log=log,
            wildcards=wildcards,
        )


def get_plot_trim_distns(wildcards):
    return expand(
        "plots/trim_distn/{manufacturer}_{metric}.png",
        manufacturer=get_all_manufacturers(wildcards),
        metric=["rank", "relprice"],
    )


rule plot_hist_odometer:
    input:
        "scripts/plot_hist_odometer.jl",
        registrations="data/arrow/registrations.arrow",
    output:
        "plots/hist_odometer.png",
    log:
        "logs/plot_hist_odometer.log",
    run:
        julia("scripts/plot_hist_odometer.jl", input=input, output=output, log=log)


rule plot_registrations_spatial:
    input:
        "scripts/plot_registrations_spatial.jl",
        new_registrations="data/arrow/new_registrations.arrow",
        dealers_init="data/dealers_init.jld2",
    output:
        "plots/registrations_spatial.png",
    log:
        "logs/plot_registrations_spatial.log",
    run:
        julia(
            "scripts/plot_registrations_spatial.jl", input=input, output=output, log=log
        )


rule plot_sales_prices_quarterly:
    input:
        "scripts/plot_sales_prices_quarterly.jl",
        new_registrations="data/arrow/new_registrations.arrow",
    output:
        "plots/sales_prices_quarterly.png",
    log:
        "logs/plot_sales_prices_quarterly.log",
    run:
        julia(
            "scripts/plot_sales_prices_quarterly.jl",
            input=input,
            output=output,
            log=log,
        )


rule plot_transaction_group_sizes:
    input:
        "scripts/plot_transaction_group_sizes.jl",
        dealers_init="data/dealers_init.jld2",
        transactions_for_analysis="data/transactions_for_analysis.jld2",
    output:
        "plots/transaction_group_sizes.png",
    log:
        "logs/plot_transaction_group_sizes.log",
    run:
        julia(
            "scripts/plot_transaction_group_sizes.jl",
            input=input,
            output=output,
            log=log,
        )


rule plot_cr4_timeline:
    input:
        "scripts/plot_cr4_timeline.jl",
        dealers_init="data/dealers_init.jld2",
    output:
        "plots/cr4_timeline.png",
    log:
        "logs/plot_cr4_timeline.log",
    run:
        julia("scripts/plot_cr4_timeline.jl", input=input, output=output, log=log)


rule table_trim_variation:
    input:
        "scripts/table_trim_variation.jl",
        price_table="data/price_table.arrow",
        new_registrations="data/arrow/new_registrations.arrow",
    output:
        "tables/trim_variation.tex",
    log:
        "logs/table_trim_variation.log",
    run:
        julia("scripts/table_trim_variation.jl", input=input, output=output, log=log)


rule table_msa_concentration:
    input:
        "scripts/table_msa_concentration.jl",
        dealers_init="data/dealers_init.jld2",
        transactions_for_analysis="data/transactions_for_analysis.jld2",
    output:
        "tables/msa_concentration.tex",
    log:
        "logs/table_msa_concentration.log",
    run:
        julia("scripts/table_msa_concentration.jl", input=input, output=output, log=log)


rule facts:
    input:
        "scripts/facts.jl",
        *julia_src,
        get_all_treatment_effects,
        new_registrations="data/arrow/new_registrations.arrow",
        price_table="data/price_table.arrow",
        popular_models="data/arrow/popular_models.arrow",
        dealers_init="data/dealers_init.jld2",
        transaction_uids="data/transaction_uids.csv",
    output:
        "facts.tex",
    log:
        "logs/facts.log",
    run:
        julia("scripts/facts.jl", input=input, output=output, log=log)


rule clean_expensive:
    """Clean the most expensive intermediate files (vins_decoded, registrations, dealers_init)"""
    run:
        expensive_files = "data/arrow/vins_decoded.arrow data/arrow/registrations.arrow data/arrow/unmatched_registrations.arrow data/arrow/popular_models.arrow data/arrow/new_registrations.arrow data/dealers_init.jld2".split()

        for f in expensive_files:
            shell(f"rm -rf {f}")


rule clean_analysis:
    """Clean analysis files but keep the basic data processing"""
    run:
        analysis_files = "data/transactions_for_analysis.jld2 data/treated data/control data/transaction_uids.csv data/affiliated_dealerships data/closest_dealerships data/related_dealerships.txt".split()

        for f in analysis_files:
            shell(f"rm -rf {f}")


rule clean:
    """Clean everything from price_table onwards (keeps vins_decoded, registrations, new_registrations, dealers_init)"""
    run:
        touched_files = "data/price_table.arrow data/treatment_effects data/group_market_share_changes.arrow data/dealer_demographics.arrow data/statistics.txt plots/event_study plots/*.png plots/trim_distn tables/*.tex facts.tex logs".split()

        for f in touched_files:
            shell(f"rm -rf {f}")


rule clean_all:
    """Clean everything (equivalent to old clean rule)"""
    run:
        all_files = "data/arrow data/dealers_init.jld2 data/price_table.arrow data/transactions_for_analysis.jld2 data/treatment_effects data/treated data/control data/transaction_uids.csv data/affiliated_dealerships data/closest_dealerships data/related_dealerships.txt data/group_market_share_changes.arrow data/dealer_demographics.arrow data/statistics.txt plots/event_study plots/*.png plots/trim_distn tables/*.tex facts.tex logs".split()

        for f in all_files:
            shell(f"rm -rf {f}")


rule all:
    input:
        get_all_treatment_effects,
        rules.dealer_group_size_histogram.output,
        rules.table_body_make_count.output,
        rules.table_te_crosstabs.output,
        rules.table_te_drilldown.output,
        get_plot_market_share_changes_tes,
        get_plot_trim_distns,
        rules.plot_hist_odometer.output,
        rules.plot_registrations_spatial.output,
        rules.plot_sales_prices_quarterly.output,
        rules.plot_cr4_timeline.output,
        rules.table_trim_variation.output,
        rules.facts.output,
        rules.plot_transaction_group_sizes.output,
        rules.table_msa_concentration.output,
        rules.table_effect_on_closest.output,


### UTILS: running julia
from snakemake.shell import shell
import tempfile
from json import dumps


def process_args(x):
    """
    Processes input/output/etc. as dict if it has kwargs or a list if it doesn't.
    """
    kwargs = dict(x)
    if kwargs:
        args_as_dict = {i + 1: v for i, v in enumerate(x)}
        return args_as_dict | kwargs
    else:
        return list(x)


def julia(scriptfile, **rule):
    """
    Run a Julia script.
    """
    rule_dict = {k: process_args(v) for k, v in rule.items()}
    # write dumps(rule) to a temp file
    # use tempfile package
    with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as f:
        f.write(dumps(rule_dict).encode())
        f.flush()
        # make sure to close the file
        f.close()
        daemon_mode = config.get("daemon-mode", "no") == "yes"
        if daemon_mode:
            shell(
                f"julia --project=. -e 'using DaemonMode; runargs()' {scriptfile} '{f.name}'"
            )
        else:
            # run the script with the temp file as input
            shell(f"julia --project=. --threads=auto {scriptfile} < '{f.name}'")
        # delete the temp file
        os.remove(f.name)
