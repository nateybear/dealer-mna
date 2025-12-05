import glob
import pandas as pd
import os
from itertools import chain

# Global variables for Julia source files
julia_src = glob.glob("src/**/*.jl")
julia_dealers = glob.glob("src/dealers/**/*.jl")
julia_empirics = glob.glob("src/empirics/**/*.jl")


# Wildcard constraints
wildcard_constraints:
    transaction_uid="[0-9a-f\\-]+",
    dealer_uid="[0-9a-f\\-]+",
    transacted_dealer_uid="[0-9a-f\\-]+",
    statistic="[a-z_]+",
    cut="(urban|exiting_seller)",
    folder="[a-z_]+",
    buy_sell="[a-z]+",
    i="[0-9]+",
    specification="(callawaysantanna|weights|none)",


# ============================================================================
# DATA GENERATION RULES
# ============================================================================


rule data_vins_decoded:
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


rule data_registrations:
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


# a "halfway" step that's necessary to avoid a cyclic dependency with dealers_init
# (dealers_init needs to know popular models, which depends on count of new_registrations,
# but new_registrations needs to match against dealers. Count popular_models from
# not-necessarily-matched new registrations)
rule data_unmatched_registrations:
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


rule data_dealers_init:
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


rule data_new_registrations:
    input:
        "scripts/new_registrations.jl",
        unmatched_registrations="data/arrow/unmatched_registrations.arrow",
        price_table="data/price_table.arrow",
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


# trick price table by using unmatched registrations instead of matched registrations
rule data_price_table:
    input:
        "scripts/price_table.jl",
        "src/registrations/normalize_price.jl",
        new_registrations="data/arrow/unmatched_registrations.arrow",
    output:
        "data/price_table.arrow",
    log:
        "logs/price_table.log",
    run:
        julia("scripts/price_table.jl", input=input, output=output, log=log)


checkpoint data_all_manufacturers:
    input:
        "scripts/all_manufacturers.jl",
        price_table="data/price_table.arrow",
    output:
        "data/all_manufacturers.txt",
    log:
        "logs/all_manufacturers.log",
    run:
        julia("scripts/all_manufacturers.jl", input=input, output=output, log=log)


# ============================================================================
# CHECKPOINT AGGREGATION HELPERS
# ============================================================================


def get_transaction_uids():
    """Get transaction UIDs from checkpoint"""
    return pd.read_csv(checkpoints.data_transaction_uids.get().output[0])


def get_related_dealer_map(wildcards):
    """Get dealer mapping files for a specific group type"""
    transaction_uids = get_transaction_uids()
    group = wildcards.group

    if group == "buyer":
        return {
            t.transaction_uid: checkpoints.data_affiliated_dealerships.get(
                transaction_uid=t.transaction_uid
            ).output.buyer_group
            for t in transaction_uids.itertuples()
        }
    elif group == "seller":
        return {
            t.transaction_uid: checkpoints.data_affiliated_dealerships.get(
                transaction_uid=t.transaction_uid
            ).output.seller_group
            for t in transaction_uids.itertuples()
        }
    elif group == "closest_franchise":
        # For closest franchise, we need to aggregate all closest dealer files for each transaction
        dealer_map = {}
        for t in transaction_uids.itertuples():
            try:
                closest_file = checkpoints.data_closest_dealerships.get(
                    transaction_uid=t.transaction_uid,
                    dealer_uid=t.dealer_uid,
                    dealer_type="franchise",
                ).output[0]
                dealer_map[t.transaction_uid] = closest_file
            except (FileNotFoundError, OSError):
                # Checkpoint file doesn't exist yet, skip for now
                pass
        return dealer_map
    elif group == "closest_any":
        # For closest any, we need to aggregate all closest dealer files for each transaction
        dealer_map = {}
        for t in transaction_uids.itertuples():
            try:
                closest_file = checkpoints.data_closest_dealerships.get(
                    transaction_uid=t.transaction_uid,
                    dealer_uid=t.dealer_uid,
                    dealer_type="any",
                ).output[0]
                dealer_map[t.transaction_uid] = closest_file
            except (FileNotFoundError, OSError):
                # Checkpoint file doesn't exist yet, skip for now
                pass
        return dealer_map
    else:
        raise ValueError(f"Unknown group type: {group}")


def get_all_manufacturers(wildcards):
    """Get the list of all manufacturers from checkpoint"""
    with open(checkpoints.data_all_manufacturers.get().output[0], "r") as f:
        return f.read().splitlines()


checkpoint data_statistics:
    input:
        "scripts/statistics.jl",
        "src/empirics/event_study/event_study.jl",
    output:
        "data/statistics.txt",
    log:
        "logs/statistics.log",
    run:
        julia("scripts/statistics.jl", output=output, log=log)


rule data_transactions_for_analysis:
    input:
        *julia_dealers,
        "src/empirics/Empirics.jl",
        "scripts/transactions_for_analysis.jl",
        dealers_init="data/dealers_init.jld2",
        price_table="data/price_table.arrow",
        new_registrations="data/arrow/new_registrations.arrow",
        popular_models="data/arrow/popular_models.arrow",
        dealer_demographics="data/dealer_demographics.arrow",
    output:
        "data/transactions_for_analysis.jld2",
    log:
        "logs/transactions_for_analysis.log",
    run:
        julia(
            "scripts/transactions_for_analysis.jl", input=input, output=output, log=log
        )


checkpoint data_transaction_uids:
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


rule data_treated_registrations:
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
        "data/treated/treated_dealers.arrow",
    log:
        "logs/treated_registrations_treated_dealers.log",
    run:
        julia(
            "scripts/treated_registrations.jl",
            input=input,
            output=output,
            log=log,
            wildcards=wildcards,
        )


rule data_control_registrations:
    input:
        "src/empirics/registrations.jl",
        "src/empirics/Empirics.jl",
        "scripts/control_registrations.jl",
        transactions_for_analysis="data/transactions_for_analysis.jld2",
        dealers_init="data/dealers_init.jld2",
        price_table="data/price_table.arrow",
        new_registrations="data/arrow/new_registrations.arrow",
        popular_models="data/arrow/popular_models.arrow",
        dealer_demographics="data/dealer_demographics.arrow",
    output:
        "data/control/treated_dealers.arrow",
    log:
        "logs/control_registrations_treated_dealers.log",
    run:
        julia(
            "scripts/control_registrations.jl",
            input=input,
            output=output,
            log=log,
            wildcards=wildcards,
        )


rule data_treated_registrations_related:
    wildcard_constraints:
        group="(buyer|seller|closest_franchise|closest_any)",
    input:
        "src/empirics/registrations.jl",
        "src/empirics/Empirics.jl",
        "scripts/treated_registrations.jl",
        transactions_for_analysis="data/transactions_for_analysis.jld2",
        dealers_init="data/dealers_init.jld2",
        price_table="data/price_table.arrow",
        new_registrations="data/arrow/new_registrations.arrow",
        popular_models="data/arrow/popular_models.arrow",
        dealer_uids=lambda w: get_related_dealer_map(w).values(),
    output:
        "data/treated/{group}.arrow",
    log:
        "logs/treated_registrations_{group}.log",
    params:
        dealer_uids=get_related_dealer_map,
    run:
        julia(
            "scripts/treated_registrations.jl",
            input=input,
            output=output,
            log=log,
            wildcards=wildcards,
            params=params,
        )


rule data_control_registrations_related:
    wildcard_constraints:
        group="(buyer|seller|closest_franchise|closest_any)",
    input:
        "src/empirics/registrations.jl",
        "src/empirics/Empirics.jl",
        "scripts/control_registrations.jl",
        transactions_for_analysis="data/transactions_for_analysis.jld2",
        dealers_init="data/dealers_init.jld2",
        price_table="data/price_table.arrow",
        new_registrations="data/arrow/new_registrations.arrow",
        popular_models="data/arrow/popular_models.arrow",
        dealer_demographics="data/dealer_demographics.arrow",
        dealer_uids=lambda w: get_related_dealer_map(w).values(),
    output:
        "data/control/{group}.arrow",
    log:
        "logs/control_registrations_{group}.log",
    params:
        dealer_uids=get_related_dealer_map,
    run:
        julia(
            "scripts/control_registrations.jl",
            input=input,
            output=output,
            log=log,
            wildcards=wildcards,
            params=params,
        )


def get_related_dealerships(wildcards):
    """Get all related dealership files"""
    transaction_uids = get_transaction_uids()
    out = set()
    for t in transaction_uids.itertuples():
        out.add(
            f"data/closest_dealerships/{t.transaction_uid}_{t.dealer_uid}_franchise.txt"
        )
        out.add(f"data/closest_dealerships/{t.transaction_uid}_{t.dealer_uid}_any.txt")
        out.add(f"data/affiliated_dealerships/buyer_group_{t.transaction_uid}.txt")
        out.add(f"data/affiliated_dealerships/seller_group_{t.transaction_uid}.txt")
    return list(out)


def get_all_related_dealerships(wildcards):
    """Get all related dealership files"""
    transaction_uids = get_transaction_uids()
    out = set()
    for t in transaction_uids.itertuples():
        out.add(f"data/affiliated_dealerships/buyer_group_{t.transaction_uid}.txt")
        out.add(f"data/affiliated_dealerships/seller_group_{t.transaction_uid}.txt")
        out.add(
            f"data/closest_dealerships/{t.transaction_uid}_{t.dealer_uid}_franchise.txt"
        )
        out.add(f"data/closest_dealerships/{t.transaction_uid}_{t.dealer_uid}_any.txt")
    return list(out)


# these generate sets of uids for each transaction
checkpoint data_affiliated_dealerships:
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
checkpoint data_closest_dealerships:
    wildcard_constraints:
        dealer_type="(franchise|any)",
        transaction_uid="[0-9a-f\\-]+",
        dealer_uid="[0-9a-f\\-]+",
    input:
        "data/transaction_uids.csv",
        "scripts/closest_dealerships.jl",
        dealers_init="data/dealers_init.jld2",
        transactions="data/transactions_for_analysis.jld2",
        new_registrations="data/arrow/new_registrations.arrow",
        popular_models="data/arrow/popular_models.arrow",
    output:
        "data/closest_dealerships/{transaction_uid}_{dealer_uid}_{dealer_type}.txt",
    log:
        "logs/closest_dealerships_{transaction_uid}_{dealer_uid}_{dealer_type}.log",
    run:
        julia(
            "scripts/closest_dealerships.jl",
            input=input,
            output=output,
            log=log,
            wildcards=wildcards,
        )


# used to force the checkpoints to run
checkpoint data_related_dealerships:
    input:
        "data/transaction_uids.csv",
        get_related_dealerships,
        new_registrations="data/arrow/new_registrations.arrow",
        popular_models="data/arrow/popular_models.arrow",
    output:
        "data/related_dealerships.txt",
    shell:
        "touch {output}"

rule data_zip_demographics:
    input:
        "scripts/zip_demographics.jl",
    output:
        "data/zip_demographics.csv",
    log:
        "logs/zip_demographics.log",
    run:
        julia("scripts/zip_demographics.jl", input=input, output=output, log=log)


rule data_dealer_demographics:
    input:
        "scripts/dealer_demographics.jl",
        "data/zip_demographics.csv",
        dealers_init="data/dealers_init.jld2",
    output:
        "data/dealer_demographics.arrow",
    log:
        "logs/dealer_demographics.log",
    run:
        julia("scripts/dealer_demographics.jl", input=input, output=output, log=log)


rule data_treatment_effect_overall:
    input:
        glob.glob("src/empirics/*.jl"),
        "scripts/treatment_effects.jl",
        dealer_demographics="data/dealer_demographics.arrow",
        dealers_init="data/dealers_init.jld2",
        new_registrations="data/arrow/new_registrations.arrow",
        popular_models="data/arrow/popular_models.arrow",
        treated_registrations="data/treated/treated_dealers.arrow",
        control_registrations="data/control/treated_dealers.arrow",
        transaction_uids="data/transaction_uids.csv",
        transactions_for_analysis="data/transactions_for_analysis.jld2",
    output:
        increasing="data/treatment_effects/overall_increasing_{specification}.jld2",
        decreasing="data/treatment_effects/overall_decreasing_{specification}.jld2",
        overall="data/treatment_effects/overall_{specification}.jld2",
    log:
        "logs/treatment_effects_overall_{specification}.log",
    run:
        julia(
            "scripts/treatment_effects.jl",
            input=input,
            output=output,
            log=log,
            wildcards=wildcards,
        )


rule data_treatment_effect_cut:
    input:
        glob.glob("src/empirics/*.jl"),
        "scripts/treatment_effects.jl",
        "data/related_dealerships.txt",  # Ensure checkpoints are run first
        dealer_demographics="data/dealer_demographics.arrow",
        dealers_init="data/dealers_init.jld2",
        new_registrations="data/arrow/new_registrations.arrow",
        popular_models="data/arrow/popular_models.arrow",
        treated_registrations="data/treated/treated_dealers.arrow",
        control_registrations="data/control/treated_dealers.arrow",
        transaction_uids="data/transaction_uids.csv",
    output:
        "data/treatment_effects/{cut}_{specification}.jld2",
    log:
        "logs/treatment_effects_{cut}_{specification}.log",
    run:
        julia(
            "scripts/treatment_effects.jl",
            input=input,
            output=output,
            log=log,
            wildcards=wildcards,
        )


rule data_treatment_effect_cuts:
    input:
        expand(
            "data/treatment_effects/{cut}_{specification}.jld2",
            cut=["urban", "exiting_seller"],
            specification=["weights", "controls"],
        ),


rule data_group_market_share_changes:
    input:
        "scripts/group_market_share_changes.jl",
        "data/statistics.txt",
        "data/related_dealerships.txt",
        "src/empirics/market_share.jl",
        get_all_related_dealerships,
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


rule data_treatment_effect_groups:
    input:
        "scripts/treatment_effect_groups.jl",
        "data/related_dealerships.txt",
        treated_buyer="data/treated/buyer.arrow",
        control_buyer="data/control/buyer.arrow",
        treated_seller="data/treated/seller.arrow",
        control_seller="data/control/seller.arrow",
        group_market_share_changes="data/group_market_share_changes.arrow",
        dealer_demographics="data/dealer_demographics.arrow",
    output:
        "data/treatment_effects/groups_{specification}.jld2",
    log:
        "logs/treatment_effects_groups_{specification}.log",
    run:
        julia(
            "scripts/treatment_effect_groups.jl",
            input=input,
            output=output,
            log=log,
            wildcards=wildcards,
        )


rule data_treatment_effect_closest:
    wildcard_constraints:
        group="(franchise|any)",
    input:
        "scripts/treatment_effect_closest.jl",
        get_related_dealerships,
        treated_registrations="data/treated/closest_{group}.arrow",
        control_registrations="data/control/closest_{group}.arrow",
        transaction_uids="data/transaction_uids.csv",
        dealers_init="data/dealers_init.jld2",
        treatment_effect_overall="data/treatment_effects/overall_{specification}.jld2",
        dealer_demographics="data/dealer_demographics.arrow",
    output:
        "data/treatment_effects/closest_{group}_{specification}.jld2",
    log:
        "logs/treatment_effect_closest_{group}_{specification}.log",
    run:
        julia(
            "scripts/treatment_effect_closest.jl",
            input=input,
            output=output,
            log=log,
            wildcards=wildcards,
        )


rule data_model_frames:
    input:
        "scripts/data_model_frames.jl",
        "data/statistics.txt",
        dealers_init="data/dealers_init.jld2",
        new_registrations="data/arrow/new_registrations.arrow",
        price_table="data/price_table.arrow",
    output:
        "data/model_frames.jld2",
    log:
        "logs/model_frames.log",
    run:
        julia("scripts/data_model_frames.jl", input=input, output=output, log=log)


rule data_facts:
    input:
        "scripts/facts.jl",
        *julia_src,
        "data/related_dealerships.txt",  # Ensure checkpoints are run first
        new_registrations="data/arrow/new_registrations.arrow",
        price_table="data/price_table.arrow",
        popular_models="data/arrow/popular_models.arrow",
        dealers_init="data/dealers_init.jld2",
        transaction_uids="data/transaction_uids.csv",
        transactions_for_analysis="data/transactions_for_analysis.jld2",
        dealer_demographics="data/dealer_demographics.arrow",
    output:
        "facts.tex",
    log:
        "logs/facts.log",
    run:
        julia("scripts/facts.jl", input=input, output=output, log=log)


rule plot_event_study_overall:
    input:
        "scripts/event_study_plot.jl",
        treatment_effect="data/treatment_effects/overall_{specification}.jld2",
    output:
        "plots/event_study/overall_{specification}_{statistic}.png",
    log:
        "logs/event_study_overall_{specification}_{statistic}.log",
    run:
        julia(
            "scripts/event_study_plot.jl",
            input=input,
            output=output,
            log=log,
            wildcards=wildcards,
        )


rule plot_diff_in_diff_cornerplot:
    input:
        "scripts/plot_diff_in_diff_cornerplot.jl",
        treatment_effect="data/treatment_effects/overall_weights.jld2",
    output:
        "plots/diff_in_diff_cornerplot.png",
    log:
        "logs/plot_diff_in_diff_cornerplot.log",
    run:
        julia(
            "scripts/plot_diff_in_diff_cornerplot.jl",
            input=input,
            output=output,
            log=log,
        )


rule plot_dealer_group_size_histogram:
    input:
        "scripts/dealer_group_size_histogram.jl",
        dealers_init="data/dealers_init.jld2",
        new_registrations="data/arrow/new_registrations.arrow",
        popular_models="data/arrow/popular_models.arrow",
    output:
        histogram="plots/dealer_group_size_histogram.png",
        cumulative="plots/dealer_group_size_cumulative.png",
    log:
        "logs/dealer_group_size_histogram.log",
    run:
        julia(
            "scripts/dealer_group_size_histogram.jl",
            input=input,
            output=output,
            log=log,
        )


rule plot_market_share_change_te:
    input:
        "scripts/plot_market_share_changes_tes.jl",
        group_market_share_changes="data/group_market_share_changes.arrow",
        treatment_effect="data/treatment_effects/groups_{specification}.jld2",
    output:
        "plots/market_share_change_te_{col}_{specification}.png",
    log:
        "logs/plot_market_share_change_te_{col}_{specification}.log",
    run:
        julia(
            "scripts/plot_market_share_changes_tes.jl",
            input=input,
            output=output,
            log=log,
            wildcards=wildcards,
        )


rule plot_market_share_changes_tes:
    input:
        expand(
            "plots/market_share_change_te_{col}_{specification}.png",
            col=[
                "mean_price",
                "median_markup",
                "mean_baseprice",
                "total_units",
                "trim_rank",
            ],
            specification=["weights"],
        ),


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
    """Get all trim distribution plots for all manufacturers"""
    return expand(
        "plots/trim_distn/{manufacturer}_{metric}.png",
        manufacturer=get_all_manufacturers(wildcards),
        metric=["rank", "relprice"],
    )


def get_treatment_effects(wildcards):
    """Get treatment effect files for a given folder"""
    return sorted(glob.glob(f"data/treatment_effects/{wildcards.folder}/**/*.arrow"))


def get_extras(wildcards):
    """Get extra statistics files for a given folder"""
    return sorted(
        glob.glob(f"data/treatment_effects/{wildcards.folder}/**/*_extras.jld2")
    )


# ============================================================================
# PLOT GENERATION RULES
# ============================================================================


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


rule plot_treated_control_overlap:
    input:
        "scripts/plot_treated_control_overlap.jl",
        treated_registrations="data/treated/treated_dealers.arrow",
        control_registrations="data/control/treated_dealers.arrow",
        dealer_demographics="data/dealer_demographics.arrow",
    output:
        "plots/treated_control_overlap_{demographic}.png",
    log:
        "logs/plot_treated_control_overlap_{demographic}.log",
    run:
        julia(
            "scripts/plot_treated_control_overlap.jl",
            input=input,
            output=output,
            wildcards=wildcards,
            log=log,
        )


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


rule plot_hhi_longitudinal_cross_sectional:
    input:
        "scripts/plot_hhi_longitudinal_cross_sectional.jl",
        dealers_init="data/dealers_init.jld2",
        new_registrations="data/arrow/new_registrations.arrow",
    output:
        longitudinal="plots/hhi_timeline.png",
        cross_sectional="plots/hhi_cross_sectional.png",
    log:
        "logs/plot_hhi_longitudinal_cross_sectional.log",
    run:
        julia(
            "scripts/plot_hhi_longitudinal_cross_sectional.jl",
            input=input,
            output=output,
            log=log,
        )


rule plot_event_study_all:
    input:
        expand(
            "plots/event_study/overall_{statistic}.png",
            statistic=["mean_price", "total_units", "total_sales", "trim_rank"],
        ),


rule plot_effect_on_closest:
    wildcard_constraints:
        group="(franchise|any)",
    input:
        "scripts/plot_effect_on_closest.jl",
        dealers_init="data/dealers_init.jld2",
        treatment_effects="data/treatment_effects/closest_{group}_weights.jld2",
        overall_effects="data/treatment_effects/overall_weights.jld2",
        transaction_uids="data/transaction_uids.csv",
    output:
        "plots/effect_on_closest_{group}.png",
    log:
        "logs/plot_effect_on_closest_{group}.log",
    run:
        julia("scripts/plot_effect_on_closest.jl", input=input, output=output, log=log)


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


rule table_pq_crosstabs:
    input:
        "scripts/table_pq_crosstabs.jl",
        treatment_effects="data/treatment_effects/overall_weights.jld2",
    output:
        "tables/pq_crosstabs.tex",
    log:
        "logs/table_pq_crosstabs.log",
    run:
        julia("scripts/table_pq_crosstabs.jl", input=input, output=output, log=log)


rule table_diff_in_diff:
    input:
        "scripts/table_diff_in_diff.jl",
        treatment_effects="data/treatment_effects/overall_weights.jld2",
    output:
        "tables/diff_in_diff.tex",
    log:
        "logs/table_diff_in_diff.log",
    run:
        julia("scripts/table_diff_in_diff.jl", input=input, output=output, log=log)


rule table_triple_diff:
    input:
        "scripts/table_triple_diff.jl",
        "data/related_dealerships.txt",  # Ensure checkpoints are run first
        treatment_effects="data/treatment_effects/{cut}_weights.jld2",
        dealer_demographics="data/dealer_demographics.arrow",
        transaction_uids="data/transaction_uids.csv",
    output:
        "tables/triple_diff_{cut}.tex",
    log:
        "logs/table_triple_diff_{cut}.log",
    run:
        julia(
            "scripts/table_triple_diff.jl",
            input=input,
            output=output,
            log=log,
            wildcards=wildcards,
        )


rule table_triple_diffs:
    input:
        expand(
            "tables/triple_diff_{cut}.tex",
            cut=["urban", "exiting_seller"],
        ),


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


rule all:
    input:
        # Tables
        "tables/body_make_count.tex",
        "tables/pq_crosstabs.tex",
        "tables/diff_in_diff.tex",
        "tables/triple_diff_urban.tex",
        "tables/triple_diff_exiting_seller.tex",
        #"tables/effect_on_closest.tex",
        "tables/trim_variation.tex",
        "tables/msa_concentration.tex",
        # Main plots
        "plots/dealer_group_size_histogram.png",
        "plots/dealer_group_size_cumulative.png",
        "plots/diff_in_diff_cornerplot.png",
        "plots/hist_odometer.png",
        "plots/registrations_spatial.png",
        "plots/sales_prices_quarterly.png",
        "plots/transaction_group_sizes.png",
        "plots/hhi_timeline.png",
        "plots/hhi_cross_sectional.png",
        expand(
            "plots/treated_control_overlap_{demographic}.png",
            demographic=[
                "total_population",
                "change_population",
                "median_income",
                "share_urban",
                "share_black",
                "share_hispanic",
            ],
        ),
        expand("plots/effect_on_closest_{group}.png", group=["franchise", "any"]),
        # Event study plots
        expand(
            [
                "plots/event_study/overall_{specification}_mean_price.png",
                "plots/event_study/overall_{specification}_total_units.png",
                "plots/event_study/overall_{specification}_total_sales.png",
                "plots/event_study/overall_{specification}_trim_rank.png",
            ],
            specification=["weights", "none", "callawaysantanna"],
        ),
        # Market share change plots
        "plots/market_share_change_te_mean_price_weights.png",
        "plots/market_share_change_te_median_markup_weights.png",
        "plots/market_share_change_te_mean_baseprice_weights.png",
        "plots/market_share_change_te_total_units_weights.png",
        "plots/market_share_change_te_trim_rank_weights.png",
        # Trim distribution plots
        get_plot_trim_distns,
        # Facts
        "facts.tex",
        # # Demand model things
        # "data/model_frames/distances.jld2",
        # "data/model_frames/agent_data.arrow",
        # "data/model_frames/product_data.arrow",
        # "data/model_frames/counts.arrow",


# ============================================================================
# UTILITY AND CLEANUP RULES
# ============================================================================


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


rule show_python_info:
    run:
        import sys

        print(sys.executable)
        print(sys.version)


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

### UTILS: running julia
from snakemake.shell import shell
import tempfile
from json import dumps
import os


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
