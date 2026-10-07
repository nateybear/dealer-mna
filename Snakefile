import glob
import pandas as pd
import os
from itertools import chain

# Global variables for Julia source files
julia_src = glob.glob("src/**/*.jl")
julia_dealers = glob.glob("src/dealers/**/*.jl")
julia_empirics = glob.glob("src/empirics/**/*.jl")


# Number of BBL forward-sim replications for the stratified pipeline. Kept
# small for fast local cuts; bump for the final run. Every stratified rule
# that iterates over seeds pulls from here.
STRATIFIED_SEEDS = range(1, 101)


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
    specification="(callawaysantanna|twfe|sunabraham|synthetic)",
    who="(treated_dealers|affiliated)",
    seed="[0-9]+",


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

rule data_dealers_init_filtered:
    input:
        "scripts/dealers_init_filtered.jl",
        new_registrations="data/arrow/new_registrations.arrow",
        dealers_init="data/dealers_init.jld2",
    output:
        "data/dealers_init_filtered.jld2",
    log:
        "logs/new_registrations.log",
    run:
        julia(
            "scripts/dealers_init_filtered.jl",
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

rule data_demand_estimates:
    input:
        "scripts/demand_estimation_dense.jl",
        model="data/model_frames.jld2",
        params_restart=lambda _: config.get("params_restart", []),
    output:
        "data/demand_estimates.jld2",
    log:
        "logs/demand_estimation.log",
    params:
        tmpdir=lambda _: config.get("tmpdir", None),
    run:
        julia(
            "scripts/demand_estimation_dense.jl",
            input=input,
            output=output,
            log=log,
            params=params,
            extra_args=["knitro", "--full", "--multistart"],
        )

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

    if group == "affiliated":
        affiliates = lambda t: checkpoints.data_affiliated_dealerships.get(
            transaction_uid=t.transaction_uid
        ).output
        return {
            t.transaction_uid: [affiliates(t).buyer_group, affiliates(t).seller_group]
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
                dealer_map[t.transaction_uid] = [closest_file]
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
                dealer_map[t.transaction_uid] = [closest_file]
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
        dealers_init="data/dealers_init_filtered.jld2",
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
        dealers_init="data/dealers_init_filtered.jld2",
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
        dealers_init="data/dealers_init_filtered.jld2",
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
        dealers_init="data/dealers_init_filtered.jld2",
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
        group="(affiliated|closest_franchise|closest_any)",
    input:
        "src/empirics/registrations.jl",
        "src/empirics/Empirics.jl",
        "scripts/treated_registrations.jl",
        transactions_for_analysis="data/transactions_for_analysis.jld2",
        dealers_init="data/dealers_init_filtered.jld2",
        price_table="data/price_table.arrow",
        new_registrations="data/arrow/new_registrations.arrow",
        popular_models="data/arrow/popular_models.arrow",
        dealer_uids=lambda w: chain.from_iterable(get_related_dealer_map(w).values()),
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
        group="(affiliated|closest_franchise|closest_any)",
    input:
        "src/empirics/registrations.jl",
        "src/empirics/Empirics.jl",
        "scripts/control_registrations.jl",
        transactions_for_analysis="data/transactions_for_analysis.jld2",
        dealers_init="data/dealers_init_filtered.jld2",
        price_table="data/price_table.arrow",
        new_registrations="data/arrow/new_registrations.arrow",
        popular_models="data/arrow/popular_models.arrow",
        dealer_demographics="data/dealer_demographics.arrow",
        dealer_uids=lambda w: chain.from_iterable(get_related_dealer_map(w).values()),
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
        "scripts/affiliated_dealerships.jl",
        transactions="data/transactions_for_analysis.jld2",
        transaction_uids="data/transaction_uids.csv",
        dealers_init="data/dealers_init_filtered.jld2",
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
        dealers_init="data/dealers_init_filtered.jld2",
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
        dealers_init="data/dealers_init_filtered.jld2",
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
        dealers_init="data/dealers_init_filtered.jld2",
        new_registrations="data/arrow/new_registrations.arrow",
        popular_models="data/arrow/popular_models.arrow",
        treated_registrations="data/treated/treated_dealers.arrow",
        control_registrations="data/control/treated_dealers.arrow",
        transaction_uids="data/transaction_uids.csv",
        transactions_for_analysis="data/transactions_for_analysis.jld2",
        market_share_changes="data/group_market_share_changes_treated_dealers.arrow",
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
        dealers_init="data/dealers_init_filtered.jld2",
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
            specification=["twfe", "sunabraham", "synthetic"],
        ),


rule data_group_market_share_changes_affiliated:
    input:
        "scripts/group_market_share_changes.jl",
        "data/statistics.txt",
        "data/related_dealerships.txt",
        "src/empirics/market_share.jl",
        get_all_related_dealerships,
        dealers_init="data/dealers_init_filtered.jld2",
        transaction_uids="data/transaction_uids.csv",
        transactions_for_analysis="data/transactions_for_analysis.jld2",
        new_registrations="data/arrow/new_registrations.arrow",
    output:
        "data/group_market_share_changes_affiliated.arrow",
    log:
        "logs/group_market_share_changes.log",
    run:
        julia(
            "scripts/group_market_share_changes.jl",
            input=input,
            output=output,
            log=log,
        )


rule data_group_market_share_changes_treated_dealers:
    input:
        "scripts/transactions_market_share_change.jl",
        "src/empirics/market_share.jl",
        dealers_init="data/dealers_init_filtered.jld2",
        transaction_uids="data/transaction_uids.csv",
        transactions_for_analysis="data/transactions_for_analysis.jld2",
        new_registrations="data/arrow/new_registrations.arrow",
    output:
        "data/group_market_share_changes_treated_dealers.arrow",
    log:
        "logs/group_market_share_changes_treated_dealers.log",
    run:
        julia(
            "scripts/transactions_market_share_change.jl",
            input=input,
            output=output,
            log=log,
        )


rule data_treatment_effect_groups:
    input:
        "scripts/treatment_effect_groups.jl",
        "data/related_dealerships.txt",
        transactions_for_analysis="data/transactions_for_analysis.jld2",
        treated="data/treated/{who}.arrow",
        control="data/control/{who}.arrow",
        group_market_share_changes="data/group_market_share_changes_{who}.arrow",
        dealer_demographics="data/dealer_demographics.arrow",
    output:
        "data/treatment_effects/groups_{who}_{specification}.jld2",
    log:
        "logs/treatment_effects_groups_{who}_{specification}.log",
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
        transactions_for_analysis="data/transactions_for_analysis.jld2",
        treated_registrations="data/treated/closest_{group}.arrow",
        control_registrations="data/control/closest_{group}.arrow",
        transaction_uids="data/transaction_uids.csv",
        dealers_init="data/dealers_init_filtered.jld2",
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
        "src/demand/loading.jl",
        dealers_init="data/dealers_init_filtered.jld2",
        registrations="data/arrow/fords.arrow",
        price_table="data/price_table.arrow",
        kept_trims="data/kept_trims.arrow",
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
        dealers_init="data/dealers_init_filtered.jld2",
        transaction_uids="data/transaction_uids.csv",
        transactions_for_analysis="data/transactions_for_analysis.jld2",
        dealer_demographics="data/dealer_demographics.arrow",
        wtp_distance="data/wtp_distance.jld2",
        mcem_trace_summary="data/mcem_trace_summary.jld2",
    output:
        "facts.tex",
    log:
        "logs/facts.log",
    run:
        julia("scripts/facts.jl", input=input, output=output, log=log)

rule data_fords:
    input:
        "scripts/fords.jl",
        new_registrations="data/arrow/new_registrations.arrow",
        dealers_init="data/dealers_init_filtered.jld2",
    output:
        "data/arrow/fords.arrow",
    log:
        "logs/fords.log",
    run:
        julia("scripts/fords.jl", input=input, output=output, log=log)


rule plot_event_study_overall:
    input:
        "scripts/event_study_plot.jl",
        treatment_effect_increasing="data/treatment_effects/overall_increasing_{specification}.jld2",
        treatment_effect_decreasing="data/treatment_effects/overall_decreasing_{specification}.jld2",
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


rule table_diff_in_diff_correlations:
    input:
        "scripts/table_diff_in_diff_correlations.jl",
        treatment_effect="data/treatment_effects/overall_{specification}.jld2",
    output:
        "tables/diff_in_diff_correlations_{specification}.tex",
    log:
        "logs/table_diff_in_diff_correlations_{specification}.log",
    run:
        julia(
            "scripts/table_diff_in_diff_correlations.jl",
            input=input,
            output=output,
            log=log,
        )


rule plot_dealer_group_size_histogram:
    input:
        "scripts/dealer_group_size_histogram.jl",
        dealers_init="data/dealers_init_filtered.jld2",
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


rule plot_hist_market_share_changes:
    input:
        "scripts/plot_hist_market_share_changes.jl",
        transactions_market_share_change="data/group_market_share_changes_treated_dealers.arrow",
    output:
        histogram="plots/hist_market_share_changes.png",
        before_after_market_share="plots/hist_market_share_changes_before_after.png",
        before_after_hhi="plots/hist_market_share_changes_before_after_hhi.png",
    log:
        "logs/hist_market_share_changes.log",
    run:
        julia(
            "scripts/plot_hist_market_share_changes.jl",
            input=input,
            output=output,
            log=log,
        )


rule plot_market_share_change_te:
    input:
        "scripts/plot_market_share_changes_tes.jl",
        group_market_share_changes="data/group_market_share_changes_{who}.arrow",
        treatment_effect="data/treatment_effects/groups_{who}_{specification}.jld2",
    output:
        "plots/market_share_change_te_{col}_{who}_{specification}.png",
    log:
        "logs/plot_market_share_change_te_{col}_{who}_{specification}.log",
    run:
        julia(
            "scripts/plot_market_share_changes_tes.jl",
            input=input,
            output=output,
            log=log,
            wildcards=wildcards,
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
        dealers_init="data/dealers_init_filtered.jld2",
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
        dealers_init="data/dealers_init_filtered.jld2",
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
        dealers_init="data/dealers_init_filtered.jld2",
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
        dealers_init="data/dealers_init_filtered.jld2",
        treatment_effect="data/treatment_effects/closest_{group}_{specification}.jld2",
        overall_effects="data/treatment_effects/overall_{specification}.jld2",
        transaction_uids="data/transaction_uids.csv",
    output:
        "plots/effect_on_closest_{group}_{specification}.png",
    log:
        "logs/plot_effect_on_closest_{group}_{specification}.log",
    run:
        julia(
            "scripts/plot_effect_on_closest.jl",
            input=input,
            output=output,
            log=log,
            wildcards=wildcards,
        )

rule data_kept_trims:
    """
    Ford (model, extra) rows whose share of dealer-quarters is above the
    THRESHOLD_SHARE cutoff in scripts/kept_trims.jl. Rare-trim filter that
    propagates through residual_demand and every supply-side booster so BBL
    forward simulation never draws a VehicleKey without a fitted density.
    """
    input:
        "scripts/kept_trims.jl",
        fords="data/arrow/fords.arrow",
    output:
        kept_trims="data/kept_trims.arrow",
    log:
        "logs/kept_trims.log",
    run:
        julia("scripts/kept_trims.jl", input=input, output=output, log=log)


rule data_residual_demand:
    """
    Per (t, f) residual-demand summary matrix: intercept + own/cross price
    coefficients per carried product. Filtered through kept_trims so rare
    Ford trims are excluded from the state features every downstream
    booster conditions on. Distributed via `-p N`; each worker holds its
    own Counterfactual so the per-(t,f) ModelDelete/undo cycle inside
    Supply.residual_demand can't collide. Hardcoded input paths; the rule
    input declaration is for dependency tracking only.
    """
    input:
        "scripts/residual_demands.jl",
        model_frames="data/model_frames.jld2",
        demand_estimates="data/demand_estimates.jld2",
        price_table="data/price_table.arrow",
    output:
        residual_demand="data/residual_demand.jld2",
    log:
        "logs/residual_demand.log",
    run:
        julia(
            "scripts/residual_demands.jl",
            input=input,
            output=output,
            log=log,
            processes=config.get("processes", 1),
        )


rule data_compliance_estimation:
    """
    Cross-fit CEM (step 1) + full-data OOS-best-round refit (step 2), with the
    sales-target booster φ(X) fit jointly each SEM round. The script uses
    hardcoded data/ paths, so the input/output declarations exist only for
    Snakemake dependency tracking. Runs with Distributed via `-p N` — set
    `--config processes=N` to choose the worker count.
    """
    input:
        "scripts/compliance_estimation.jl",
        dealers_init="data/dealers_init_filtered.jld2",
        fords="data/arrow/fords.arrow",
        dealer_demographics="data/dealer_demographics.arrow",
        model_frames="data/model_frames.jld2",
        residual_demand="data/residual_demand.jld2",
    output:
        compliance="data/compliance.arrow",
        fits="data/compliance_fits.jld2",
        fits_final="data/compliance_fits_final.jld2",
        sales_targets="data/sales_targets.jld2",
        mcem_trace="data/compliance_mcem_trace.jld2",
    log:
        "logs/compliance_estimation.log",
    run:
        julia(
            "scripts/compliance_estimation.jl",
            input=input,
            output=output,
            log=log,
            processes=config.get("processes", 1),
        )


rule data_price_density_estimation:
    """
    Per-VehicleKey XGBoost mean and stddev boosters with OOS best-round selection.
    Now conditions on (state_{t, f}, log φ_{t, f}) — sales-target booster
    predictions per (t, f). Hardcoded input paths in the script; rule inputs
    exist for dependency tracking.
    """
    input:
        "scripts/price_density_estimation.jl",
        model_frames="data/model_frames.jld2",
        residual_demand="data/residual_demand.jld2",
        sales_targets="data/sales_targets.jld2",
        compliance="data/compliance.arrow",
        price_table="data/price_table.arrow",
    output:
        price_densities="data/price_densities.jld2",
    log:
        "logs/price_density_estimation.log",
    run:
        julia(
            "scripts/price_density_estimation.jl",
            input=input,
            output=output,
            log=log,
            processes=config.get("processes", 1),
        )


rule data_state_transition_estimation:
    """
    Per-VehicleKey (α₀, α_own, α_cross) XGBoost boosters for the state
    transition s_t = g(s_{t-1}). Follows the p_base/ρ̃ pattern: hardcoded
    input paths, Distributed via `-p N`.
    """
    input:
        "scripts/state_transition_estimation.jl",
        model_frames="data/model_frames.jld2",
        residual_demand="data/residual_demand.jld2",
        price_table="data/price_table.arrow",
    output:
        state_transition="data/state_transition.jld2",
    log:
        "logs/state_transition_estimation.log",
    run:
        julia(
            "scripts/state_transition_estimation.jl",
            input=input,
            output=output,
            log=log,
            processes=config.get("processes", 1),
        )


rule data_bbl_simulation:
    """
    One BBL forward-simulation sample path from Q1 2017 through Q4 2019.
    RNG seeded via the `seed` wildcard so Snakemake can fan out across
    seeds in parallel. Consumes every supply-side estimator + demand
    counterfactual machinery.
    """
    input:
        "scripts/bbl_forward_simulation.jl",
        model_frames="data/model_frames.jld2",
        demand_estimates="data/demand_estimates.jld2",
        delta_decomp="data/delta_decomposition_regout.jld2",
        compliance_fits="data/compliance_fits_final.jld2",
        sales_targets="data/sales_targets.jld2",
        price_densities="data/price_densities.jld2",
        state_transition="data/state_transition.jld2",
        compliance="data/compliance.arrow",
        residual_demand="data/residual_demand.jld2",
        price_table="data/price_table.arrow",
    output:
        sim="data/bbl_simulation/seed_{seed}.jld2",
    log:
        "logs/bbl_forward_simulation_seed_{seed}.log",
    run:
        julia(
            "scripts/bbl_forward_simulation.jl",
            input=input,
            output=output,
            wildcards=wildcards,
            log=log,
        )

rule data_bbl_simulations:
    input:
        expand("data/bbl_simulation/seed_{seed}.jld2", seed=range(1, 201))


rule data_bbl_matrix:
    """
    Aggregates the BBL sim seeds into BBLMatrices — one (R, C) pair per
    (FE type, scenario, seed) triple. Rebuilds the same eviction / usable_vks
    pruning as bbl_forward_simulation so the BBLIndex's dealer + vehicle
    columns align with the sim outputs 1:1.
    """
    input:
        "scripts/bbl_matrix.jl",
        model_frames="data/model_frames.jld2",
        demand_estimates="data/demand_estimates.jld2",
        delta_decomp="data/delta_decomposition_regout.jld2",
        price_densities="data/price_densities.jld2",
        state_transition="data/state_transition.jld2",
        simulations=expand("data/bbl_simulation/seed_{seed}.jld2", seed=range(1, 201)),
    output:
        "data/bbl_matrix.jld2",
    log:
        "logs/bbl_matrix.log",
    run:
        julia(
            "scripts/bbl_matrix.jl",
            input=input,
            output=output,
            log=log,
        )


rule data_bbl_matrix_maps:
    """
    Flatten BBLMatrices into memory-mapped Float64 arrays holding
    (observed − perturbation) diffs, one row per (scenario, tf), one column per
    sim. bbl_estimation.jl reads these via `Mmap.mmap` and hardcoded paths.
    """
    input:
        "scripts/bbl_matrix_to_map.jl",
        bbl_matrix="data/bbl_matrix.jld2",
    output:
        R="data/bbl_R.map",
        C_j="data/bbl_C_j.map",
        C_f="data/bbl_C_f.map",
        C_t="data/bbl_C_t.map",
    log:
        "logs/bbl_matrix_to_map.log",
    run:
        julia(
            "scripts/bbl_matrix_to_map.jl",
            input=input,
            output=output,
            log=log,
        )


rule data_bbl_estimation:
    """
    Solve the moment-inequality LP for identified sets on (w, f, t). Consumes
    the flat .map files produced by data_bbl_matrix_maps.
    """
    input:
        "scripts/bbl_estimation.jl",
        bbl_matrix="data/bbl_matrix.jld2",
        R="data/bbl_R.map",
        C_j="data/bbl_C_j.map",
        C_f="data/bbl_C_f.map",
        C_t="data/bbl_C_t.map",
    output:
        "data/bbl_estimates.jld2",
    log:
        "logs/bbl_estimation.log",
    run:
        julia(
            "scripts/bbl_estimation.jl",
            input=input,
            output=output,
            log=log,
        )


rule table_assortment_probs:
    input:
        "scripts/plot_assortment_probs.jl",
        fords = "data/arrow/fords.arrow",
        dealer_demographics = "data/dealer_demographics.arrow",
    output:
        reg_table="tables/markup_trim_regression.tex",
        trim_csv="tables/trim_variation_dealer_quarter.csv",
    log:
        "logs/plot_assortment_probs.log",
    run:
        julia(
            "scripts/plot_assortment_probs.jl",
            input=input,
            output=output,
            log=log,
        )


rule plot_sales_targets_scatter:
    input:
        "scripts/plot_sales_targets_scatter.jl",
        model_frames="data/model_frames.jld2",
        state_matrix="data/residual_demand.jld2",
        compliance="data/compliance.arrow",
        sales_targets="data/sales_targets.jld2",
        fords="data/arrow/fords.arrow",
    output:
        plot="plots/sales_targets_scatter.png",
    log:
        "logs/plot_sales_targets_scatter.log",
    run:
        julia(
            "scripts/plot_sales_targets_scatter.jl",
            input=input,
            output=output,
            log=log,
        )

rule plot_trim_probabilities:
    input:
        "scripts/plot_trim_probabilities.jl",
        state_matrix="data/residual_demand.jld2",
        fits="data/compliance_fits_final.jld2",
        fords="data/arrow/fords.arrow",
    output:
        popular_plot="plots/trim_probabilities_popular.png",
        variation_plot="plots/trim_probabilities_variation.png",
    log:
        "logs/plot_trim_probabilities.log",
    run:
        julia(
            "scripts/plot_trim_probabilities.jl",
            input=input,
            output=output,
            log=log,
        )

rule table_trim_probabilities:
    input:
        "scripts/table_trim_probabilities.jl",
        state_matrix="data/residual_demand.jld2",
        fits="data/compliance_fits_final.jld2",
        fords="data/arrow/fords.arrow",
    output:
        tex="tables/trim_probabilities.tex",
        csv="tables/trim_probabilities.csv",
    log:
        "logs/table_trim_probabilities.log",
    run:
        julia(
            "scripts/table_trim_probabilities.jl",
            input=input,
            output=output,
            log=log,
        )

rule plot_mcem_trace:
    input:
        "scripts/plot_mcem_trace.jl",
        mcem_trace="data/compliance_mcem_trace.jld2",
    output:
        plot="plots/mcem_trace.png",
        summary="data/mcem_trace_summary.jld2",
    log:
        "logs/plot_mcem_trace.log",
    run:
        julia(
            "scripts/plot_mcem_trace.jl",
            input=input,
            output=output,
            log=log,
        )


# ============================================================================
# STRATIFIED-FOLD PARALLEL PIPELINE
# ----------------------------------------------------------------------------
# Mirrors compliance → price densities → BBL sim → matrix → maps → estimation
# and every downstream compliance/BBL plot/table, all under `_stratified`
# suffixed sibling paths. Lets the current pipeline stay live while the
# stratified re-estimation cooks; swap in with `mv` when validated.
# See plans/compliance_stratified_folds.md for the design + swap procedure.
# ============================================================================


rule data_compliance_estimation_stratified:
    """
    Parallel-track compliance estimation with per-trim K-fold splits stratified
    on is_sold. Same script structure as data_compliance_estimation; outputs
    are written under _stratified sibling paths so both estimates coexist.
    """
    input:
        "scripts/compliance_estimation_stratified.jl",
        dealers_init="data/dealers_init_filtered.jld2",
        fords="data/arrow/fords.arrow",
        dealer_demographics="data/dealer_demographics.arrow",
        model_frames="data/model_frames.jld2",
        residual_demand="data/residual_demand.jld2",
    output:
        compliance="data/compliance_stratified.arrow",
        fits="data/compliance_fits_stratified.jld2",
        fits_final="data/compliance_fits_final_stratified.jld2",
        sales_targets="data/sales_targets_stratified.jld2",
        mcem_trace="data/compliance_mcem_trace_stratified.jld2",
    log:
        "logs/compliance_estimation_stratified.log",
    run:
        julia(
            "scripts/compliance_estimation_stratified.jl",
            input=input,
            output=output,
            log=log,
            processes=config.get("processes", 1),
        )


rule data_price_density_estimation_stratified:
    """
    Nonparametric conditional log-price density via quantile-regression grid
    (Supply.QUANTILE_LEVELS). Output is Dict{VehicleKey, Vector{Vector{UInt8}}}
    — one saved booster per quantile per VK. BBL sim samples via
    Supply.sample_from_quantile_grid (inverse-CDF with linear interpolation).
    Replaces the mean+var Normal density on the un-suffixed rule.
    """
    input:
        "scripts/price_density_estimation_stratified.jl",
        model_frames="data/model_frames.jld2",
        residual_demand="data/residual_demand.jld2",
        sales_targets="data/sales_targets_stratified.jld2",
        compliance="data/compliance_stratified.arrow",
        price_table="data/price_table.arrow",
    output:
        price_densities="data/price_densities_stratified.jld2",
    log:
        "logs/price_density_estimation_stratified.log",
    run:
        julia(
            "scripts/price_density_estimation_stratified.jl",
            input=input,
            output=output,
            log=log,
            processes=config.get("processes", 1),
        )


rule data_state_transition_estimation_stratified:
    """
    Quantile-regression grid variant of data_state_transition_estimation.
    Fits one booster per level in Supply.QUANTILE_LEVELS for each of the
    intercept, own, and cross transition targets. Same input universe as the
    un-suffixed rule (no compliance dependency) but a distinct output file
    because the format is different (Vector{Vector{UInt8}}³ vs the 6-booster
    mean/var tuple).
    """
    input:
        "scripts/state_transition_estimation_stratified.jl",
        model_frames="data/model_frames.jld2",
        residual_demand="data/residual_demand.jld2",
        price_table="data/price_table.arrow",
    output:
        state_transition="data/state_transition_stratified.jld2",
    log:
        "logs/state_transition_estimation_stratified.log",
    run:
        julia(
            "scripts/state_transition_estimation_stratified.jl",
            input=input,
            output=output,
            log=log,
            processes=config.get("processes", 1),
        )


rule data_bbl_simulation_stratified:
    """
    Stratified-branch BBL forward simulation. Uses the copied
    bbl_forward_simulation_stratified.jl (not the un-suffixed script) because
    the price + state-transition density formats differ (quantile grid vs
    mean+var). Also reads the _stratified compliance/sales_targets/
    price_densities/state_transition inputs.
    """
    input:
        "scripts/bbl_forward_simulation_stratified.jl",
        model_frames="data/model_frames.jld2",
        demand_estimates="data/demand_estimates.jld2",
        delta_decomp="data/delta_decomposition_regout.jld2",
        compliance_fits="data/compliance_fits_final_stratified.jld2",
        sales_targets="data/sales_targets_stratified.jld2",
        price_densities="data/price_densities_stratified.jld2",
        state_transition="data/state_transition_stratified.jld2",
        compliance="data/compliance_stratified.arrow",
        residual_demand="data/residual_demand.jld2",
        price_table="data/price_table.arrow",
    output:
        sim="data/bbl_simulation_stratified/seed_{seed}.jld2",
    log:
        "logs/bbl_forward_simulation_stratified_seed_{seed}.log",
    run:
        julia(
            "scripts/bbl_forward_simulation_stratified.jl",
            input=input,
            output=output,
            wildcards=wildcards,
            log=log,
        )


rule data_bbl_simulations_stratified:
    input:
        expand("data/bbl_simulation_stratified/seed_{seed}.jld2", seed=STRATIFIED_SEEDS)


rule table_diagnose_sim_assortments_stratified:
    """
    Compares simulated assortments against empirical assortments at (t, f).
    Three outputs: per-quarter size means (real vs sim), per-group marginal
    carry rates (real vs sim), and pairwise co-carry excess-over-independence
    (real vs sim). The last is the diagnostic for whether independent
    Bernoulli group-firing loses joint structure present in the data.
    """
    input:
        "scripts/diagnose_sim_assortments_stratified.jl",
        model_frames="data/model_frames.jld2",
        price_table="data/price_table.arrow",
        simulations=expand("data/bbl_simulation_stratified/seed_{seed}.jld2", seed=STRATIFIED_SEEDS),
    output:
        sizes="tables/diagnose_sim_assortments_sizes_stratified.arrow",
        groups="tables/diagnose_sim_assortments_groups_stratified.arrow",
        pairs="tables/diagnose_sim_assortments_pairs_stratified.arrow",
    log:
        "logs/diagnose_sim_assortments_stratified.log",
    run:
        julia(
            "scripts/diagnose_sim_assortments_stratified.jl",
            input=input,
            output=output,
            log=log,
        )


rule table_diagnose_booster_splits_compliance_stratified:
    """
    Inspects every split node in the distilled compliance boosters and
    reports whether the numeric threshold is trivial on non-missing feature
    values (i.e. the split is really a categorical "is this cell missing?"
    partition). High frac_trivial means the sparsity pattern of the state
    matrix is doing more work than the feature values, and increasing
    max_depth just buys more missingness partitions.
    """
    input:
        "scripts/diagnose_booster_splits_compliance_stratified.jl",
        compliance_fits_final="data/compliance_fits_final_stratified.jld2",
        residual_demand="data/residual_demand.jld2",
    output:
        detailed="tables/diagnose_booster_splits_compliance_stratified_detailed.arrow",
        summary="tables/diagnose_booster_splits_compliance_stratified_summary.arrow",
    log:
        "logs/diagnose_booster_splits_compliance_stratified.log",
    run:
        julia(
            "scripts/diagnose_booster_splits_compliance_stratified.jl",
            input=input,
            output=output,
            log=log,
        )


rule table_diagnose_state_anticipation_stratified:
    """
    Diagnostic on the "correct anticipation + mean-zero shock" model:
    scans every forward-sim seed and reports mean(state_actual - state_expected)
    per (quarter × cell class). Passes if |mean_eta| ≲ SE overall and per
    subgroup — otherwise the transition draw and residual_demand fit have a
    systematic offset that leaks into state_prev carry-forward.
    """
    input:
        "scripts/diagnose_state_anticipation_stratified.jl",
        simulations=expand("data/bbl_simulation_stratified/seed_{seed}.jld2", seed=STRATIFIED_SEEDS),
    output:
        diagnostic="tables/diagnose_state_anticipation_stratified.arrow",
    log:
        "logs/diagnose_state_anticipation_stratified.log",
    run:
        julia(
            "scripts/diagnose_state_anticipation_stratified.jl",
            input=input,
            output=output,
            log=log,
        )


rule data_bbl_matrix_stratified:
    """
    Reuses scripts/bbl_matrix.jl unchanged: it only reads the KEYS of
    price_densities and state_transition (for eviction / usable_vks), never
    the value format, so it works with both the quantile-grid stratified
    outputs and the un-suffixed mean+var outputs.
    """
    input:
        "scripts/bbl_matrix.jl",
        model_frames="data/model_frames.jld2",
        demand_estimates="data/demand_estimates.jld2",
        delta_decomp="data/delta_decomposition_regout.jld2",
        price_densities="data/price_densities_stratified.jld2",
        state_transition="data/state_transition_stratified.jld2",
        simulations=expand("data/bbl_simulation_stratified/seed_{seed}.jld2", seed=STRATIFIED_SEEDS),
    output:
        "data/bbl_matrix_stratified.jld2",
    log:
        "logs/bbl_matrix_stratified.log",
    run:
        julia(
            "scripts/bbl_matrix.jl",
            input=input,
            output=output,
            log=log,
        )


rule data_bbl_matrix_maps_stratified:
    input:
        "scripts/bbl_matrix_to_map.jl",
        bbl_matrix="data/bbl_matrix_stratified.jld2",
    output:
        R="data/bbl_R_stratified.map",
        C_j="data/bbl_C_j_stratified.map",
        C_f="data/bbl_C_f_stratified.map",
        C_t="data/bbl_C_t_stratified.map",
    log:
        "logs/bbl_matrix_to_map_stratified.log",
    run:
        julia(
            "scripts/bbl_matrix_to_map.jl",
            input=input,
            output=output,
            log=log,
        )


rule data_bbl_estimation_stratified:
    """
    bbl_estimation_stratified.jl hardcodes the _stratified matrix + map paths
    (mirrors bbl_estimation.jl's hardcoded reads via Mmap.mmap). Also reads
    data/arrow/fords.arrow to compute sales-share weights for the best/worst
    counterfactual solves, and data/model_frames.jld2 for price_scale (the
    dollars-to-std-price divisor used at demand-estimation load time).
    """
    input:
        "scripts/bbl_estimation_stratified.jl",
        bbl_matrix="data/bbl_matrix_stratified.jld2",
        R="data/bbl_R_stratified.map",
        C_j="data/bbl_C_j_stratified.map",
        C_f="data/bbl_C_f_stratified.map",
        C_t="data/bbl_C_t_stratified.map",
        fords="data/arrow/fords.arrow",
        model_frames="data/model_frames.jld2",
    output:
        "data/bbl_estimates_stratified.jld2",
    log:
        "logs/bbl_estimation_stratified.log",
    run:
        julia(
            "scripts/bbl_estimation_stratified.jl",
            input=input,
            output=output,
            log=log,
        )


rule data_nash_bertrand_counterfactual_stratified:
    """
    Nash-Bertrand price equilibrium under best_case and worst_case marginal-cost
    scenarios from the BBL identified set. Repricing is restricted to observed
    (t, j, f) triples in the 105-VK / 214-dealer BBL universe; the gap between
    equilibrium and observed prices isolates the dynamic sales-target incentive
    effect on markups.
    """
    input:
        "scripts/nash_bertrand_counterfactual.jl",
        model_frames="data/model_frames.jld2",
        demand_estimates="data/demand_estimates.jld2",
        delta_decomp="data/delta_decomposition_regout.jld2",
        bbl_estimates="data/bbl_estimates_stratified.jld2",
        bbl_matrix="data/bbl_matrix_stratified.jld2",
        price_densities="data/price_densities_stratified.jld2",
        state_transition="data/state_transition_stratified.jld2",
    output:
        counterfactual="data/nash_bertrand_counterfactual_stratified.jld2",
    log:
        "logs/nash_bertrand_counterfactual_stratified.log",
    run:
        julia(
            "scripts/nash_bertrand_counterfactual.jl",
            input=input,
            output=output,
            log=log,
        )


rule plot_sales_targets_scatter_stratified:
    input:
        "scripts/plot_sales_targets_scatter.jl",
        model_frames="data/model_frames.jld2",
        state_matrix="data/residual_demand.jld2",
        compliance="data/compliance_stratified.arrow",
        sales_targets="data/sales_targets_stratified.jld2",
        fords="data/arrow/fords.arrow",
    output:
        plot="plots/sales_targets_scatter_stratified.png",
    log:
        "logs/plot_sales_targets_scatter_stratified.log",
    run:
        julia(
            "scripts/plot_sales_targets_scatter.jl",
            input=input,
            output=output,
            log=log,
        )


rule plot_trim_probabilities_stratified:
    input:
        "scripts/plot_trim_probabilities.jl",
        state_matrix="data/residual_demand.jld2",
        fits="data/compliance_fits_final_stratified.jld2",
        fords="data/arrow/fords.arrow",
    output:
        popular_plot="plots/trim_probabilities_popular_stratified.png",
        variation_plot="plots/trim_probabilities_variation_stratified.png",
    log:
        "logs/plot_trim_probabilities_stratified.log",
    run:
        julia(
            "scripts/plot_trim_probabilities.jl",
            input=input,
            output=output,
            log=log,
        )


rule table_trim_probabilities_stratified:
    input:
        "scripts/table_trim_probabilities.jl",
        state_matrix="data/residual_demand.jld2",
        fits="data/compliance_fits_final_stratified.jld2",
        fords="data/arrow/fords.arrow",
    output:
        tex="tables/trim_probabilities_stratified.tex",
        csv="tables/trim_probabilities_stratified.csv",
    log:
        "logs/table_trim_probabilities_stratified.log",
    run:
        julia(
            "scripts/table_trim_probabilities.jl",
            input=input,
            output=output,
            log=log,
        )


rule plot_mcem_trace_stratified:
    input:
        "scripts/plot_mcem_trace.jl",
        mcem_trace="data/compliance_mcem_trace_stratified.jld2",
    output:
        plot="plots/mcem_trace_stratified.png",
        summary="data/mcem_trace_summary_stratified.jld2",
    log:
        "logs/plot_mcem_trace_stratified.log",
    run:
        julia(
            "scripts/plot_mcem_trace.jl",
            input=input,
            output=output,
            log=log,
        )


rule all_stratified:
    """
    Convenience aggregator: builds every stratified artifact the JMP paper
    uses, plus the newest data-only outputs (BBL estimates + Nash–Bertrand
    counterfactual) whose plots are not yet in the paper. The manual
    `trim_probabilities_featured_stratified.png` is produced by hand and
    is intentionally omitted.
    """
    input:
        # Paper artifacts
        "tables/trim_probabilities_stratified.tex",
        "plots/mcem_trace_stratified.png",
        "plots/sales_targets_scatter_stratified.png",
        "plots/trim_probabilities_popular_stratified.png",
        "plots/trim_probabilities_variation_stratified.png",
        # New data-only outputs (plots not yet in paper)
        "data/bbl_estimates_stratified.jld2",
        "data/nash_bertrand_counterfactual_stratified.jld2",


rule plot_wtp_distance:
    input:
        "scripts/plot_wtp_distance.jl",
        model_frames="data/model_frames.jld2",
        demand_estimates="data/demand_estimates.jld2",
        delta_decomp="data/delta_decomposition_regout.jld2",
        fords="data/arrow/fords.arrow",
    output:
        plot="plots/wtp_distance.png",
        summary="data/wtp_distance.jld2",
    log:
        "logs/plot_wtp_distance.log",
    run:
        julia(
            "scripts/plot_wtp_distance.jl",
            input=input,
            output=output,
            log=log,
        )

rule toy_model:
    input:
        "scripts/toy_model.jl",
    output:
        jld2="data/toy_model.jld2",
        # png_c_L_c_H="plots/toy_model_c_L_c_H.png",
        # png_v_L_v_H="plots/toy_model_v_L_v_H.png",
        # png_c_L_v_L="plots/toy_model_c_L_v_L.png",
        # png_c_H_v_H="plots/toy_model_c_H_v_H.png",
        profit_plot="plots/toy_model_profit.png",
        surplus_plot="plots/toy_model_surplus.png",
        mu_plot="plots/toy_model_mu.png",
        lambda_plot="plots/toy_model_lambda.png",
        wL_plot="plots/toy_model_wL.png",
        piR_plot="plots/toy_model_piR.png",
        qH_plot="plots/toy_model_qH.png",
        wH_plot="plots/toy_model_wH.png",
        # png_compliance_c_L_c_H="plots/toy_model_compliance_c_L_c_H.png",
        # png_compliance_v_L_v_H="plots/toy_model_compliance_v_L_v_H.png",
        # png_compliance_c_L_v_L="plots/toy_model_compliance_c_L_v_L.png",
        # png_compliance_c_H_v_H="plots/toy_model_compliance_c_H_v_H.png",
        compliance_wL_plot="plots/toy_model_compliance_wL.png",
        compliance_wH_plot="plots/toy_model_compliance_wH.png",
        compliance_prcomply_plot="plots/toy_model_compliance_prcomply.png",
        compliance_phi_plot="plots/toy_model_compliance_phi.png",

    log:
        "logs/toy_model.log",
    run:
        julia("scripts/toy_model.jl", input=input, output=output, log=log)


rule table_demand_standard_errors:
    input:
        "scripts/demand_standard_errors.jl",
        model="data/model_frames.jld2",
        estimates="data/demand_estimates.jld2",
    output:
        "tables/demand_standard_errors.csv",
    log:
        "logs/demand_standard_errors.log",
    run:
        julia(
            "scripts/demand_standard_errors.jl",
            input=input,
            output=output,
            log=log,
        )


rule table_diff_in_diff:
    input:
        "scripts/table_diff_in_diff.jl",
        treatment_effects=expand(
            "data/treatment_effects/overall_{direction}_{specification}.jld2",
            specification=["twfe", "sunabraham", "synthetic"],
            direction=["increasing", "decreasing"],
        ),
    output:
        "tables/diff_in_diff.tex",
    log:
        "logs/table_diff_in_diff.log",
    run:
        julia("scripts/table_diff_in_diff.jl", input=input, output=output, log=log)


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
        dealers_init="data/dealers_init_filtered.jld2",
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
        "tables/trim_variation.tex",
        "tables/msa_concentration.tex",
        "tables/diff_in_diff.tex",
        # Main plots
        "plots/dealer_group_size_histogram.png",
        "plots/dealer_group_size_cumulative.png",
        expand(
            "tables/diff_in_diff_correlations_{specification}.tex",
            specification=["twfe", "sunabraham", "synthetic"],
        ),
        "plots/hist_odometer.png",
        "plots/registrations_spatial.png",
        "plots/sales_prices_quarterly.png",
        "plots/transaction_group_sizes.png",
        "plots/hhi_timeline.png",
        "plots/hhi_cross_sectional.png",
        "plots/hist_market_share_changes.png",
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
        expand(
            "plots/effect_on_closest_{group}_{specification}.png",
            group=["franchise", "any"],
            specification=["twfe", "synthetic"],
        ),
        # Event study plots
        expand(
            "plots/event_study/overall_{specification}_{statistic}.png",
            specification=["sunabraham", "twfe", "callawaysantanna", "synthetic"],
            statistic=[
                "mean_price",
                "total_units",
                "total_sales",
                "trim_rank",
                "unique_trims",
                "median_markup",
                "mean_baseprice",
            ],
        ),
        # Market share change plots
        expand(
            "plots/market_share_change_te_{statistic}_{who}_{specification}.png",
            specification=["twfe", "synthetic"],
            who=["affiliated", "treated_dealers"],
            statistic=[
                "mean_baseprice",
                "mean_price",
                "median_markup",
                "total_units",
                "total_sales",
                "trim_rank",
                "unique_trims",
            ],
        ),
        # Trim distribution plots
        get_plot_trim_distns,
        # Assortment probability plots and trim-variation CSV
        "plots/assortment_probs_ford",
        "tables/trim_variation_dealer_quarter.csv",
        # Supply-side XGBoost estimations
        "data/compliance_fits_final.jld2",
        "data/price_densities.jld2",
        "data/sales_targets.jld2",
        # Results-section artifacts
        "plots/sales_targets_scatter.png",
        "plots/trim_probabilities_featured.png",
        "tables/trim_probabilities.tex",
        "plots/mcem_trace.png",
        "plots/wtp_distance.png",
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
        return {**args_as_dict, **kwargs}
    else:
        return list(x)


def julia(scriptfile, extra_args=None, processes=None, **rule):
    """
    Run a Julia script. `extra_args` is a list of CLI tokens spliced after the script
    name; the script reads the snakemake JSON from stdin (non-MPI, non-daemon mode).

    `processes`: when set, adds `-p N` before the script name so the worker pool is
    spun up for scripts that use Distributed. May also be supplied at invocation time
    via `--config processes=N` (this argument wins).
    """
    rule_dict = {k: process_args(v) for k, v in rule.items()}
    args_str = " ".join(extra_args) if extra_args else ""
    n_proc = config.get("processes", processes)
    p_flag = f"-p {n_proc}" if n_proc else ""
    # write dumps(rule) to a temp file
    # use tempfile package
    tmpdir = rule.get("params", {}).get("tmpdir", None)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".json", dir=tmpdir) as f:
        f.write(dumps(rule_dict).encode())
        f.flush()
        # make sure to close the file
        f.close()
        daemon_mode = config.get("daemon-mode", "no") == "yes"
        if daemon_mode:
            shell(
                f"julia --project=. -e 'using DaemonMode; runargs()' {scriptfile} {args_str} '{f.name}'"
            )
        else:
            threads = config.get("threads", "auto")
            # run the script with the temp file as input
            if config.get("mpi", ""):
                cmd = f"{config['mpi']} julia --project=. --threads={threads} {p_flag} {scriptfile} {args_str} '{f.name}'"
            else:
                cmd = f"julia --project=. --threads={threads} {p_flag} {scriptfile} {args_str} < '{f.name}'"
            shell(cmd)
        # delete the temp file
        os.remove(f.name)
