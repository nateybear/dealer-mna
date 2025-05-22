import glob
import pandas as pd

julia_src = glob.glob("src/**/*.jl")
julia_dealers = glob.glob("src/dealers/**/*.jl")
julia_empirics = glob.glob("src/empirics/**/*.jl")


wildcard_constraints:
    transaction_uid="[0-9a-f\\-]+",
    dealer_uid="[0-9a-f\\-]+",
    statistic="[a-z_]+",
    cut="[a-z_]+",


rule show_python_version:
    run:
        import sys

        print(sys.executable)
        print(sys.version)


rule dealers_init:
    input:
        *julia_dealers,
        "src/dealers/Dealers.jl",
        "scripts/dealers_init.jl",
    output:
        "data/dealers_init.jld2",
    log:
        "logs/dealers_init.log",
    run:
        julia("scripts/dealers_init.jl", output=output, log=log)


rule price_table:
    input:
        "scripts/price_table.jl",
    output:
        "data/price_table.arrow",
    log:
        "logs/price_table.log",
    run:
        julia("scripts/price_table.jl", output=output, log=log)


rule transactions_for_analysis:
    input:
        *julia_dealers,
        *julia_empirics,
        "src/empirics/Empirics.jl",
        "scripts/transactions_for_analysis.jl",
        dealers_init="data/dealers_init.jld2",
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
        transactions_for_analysis="data/transactions_for_analysis.jld2",
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
    output:
        before="data/treated_registrations/last_registrations_before_{uid}.arrow",
        after="data/treated_registrations/first_registrations_after_{uid}.arrow",
    log:
        "logs/treated_registrations_{uid}.log",
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
    output:
        before="data/control_registrations/last_registrations_before_{uid}.arrow",
        after="data/control_registrations/first_registrations_after_{uid}.arrow",
    log:
        "logs/control_registrations_{uid}.log",
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
        *julia_src,
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
        *julia_src,
        "scripts/treatment_effects.jl",
        treated_before="data/treated_registrations/last_registrations_before_{transaction_uid}.arrow",
        treated_after="data/treated_registrations/first_registrations_after_{transaction_uid}.arrow",
        control_before="data/control_registrations/last_registrations_before_{transaction_uid}.arrow",
        control_after="data/control_registrations/first_registrations_after_{transaction_uid}.arrow",
        dealer_demographics="data/dealer_demographics.arrow",
        dealers_init="data/dealers_init.jld2",
    output:
        "data/treatment_effects/{transaction_uid}/{dealer_uid}.arrow",
    log:
        "logs/treatment_effects_{transaction_uid}_{dealer_uid}.log",
    run:
        julia(
            "scripts/treatment_effects.jl",
            input=input,
            output=output,
            log=log,
            wildcards=wildcards,
        )


def get_treatment_effects(wildcards):
    # Get the output file from the checkpoint at runtime
    checkpoint_output = checkpoints.transaction_uids.get().output[0]
    uids = pd.read_csv(checkpoint_output)
    return [
        f"data/treatment_effects/{x.transaction_uid}/{x.dealer_uid}.arrow"
        for x in uids.itertuples()
    ]


checkpoint nonempty_treatment_effects:
    input:
        get_treatment_effects,
        "scripts/nonempty_treatment_effects.jl",
    output:
        "data/nonempty_treatment_effects.csv",
    log:
        "logs/nonempty_treatment_effects.log",
    run:
        julia(
            "scripts/nonempty_treatment_effects.jl",
            input=input,
            output=output,
            log=log,
            wildcards=wildcards,
        )


def get_nonempty_treatment_effects(wildcards):
    # Get the output file from the checkpoint at runtime
    transaction_uids = pd.read_csv(
        checkpoints.nonempty_treatment_effects.get().output[0]
    )
    return [
        f"data/treatment_effects/{x.transaction_uid}/{x.dealer_uid}.arrow"
        for x in transaction_uids.itertuples()
    ]


rule event_study_plot:
    input:
        "data/treatment_effects/{transaction_uid}/{dealer_uid}.arrow",
        "scripts/event_study_plot.jl",
    output:
        "plots/event_study/{statistic}/treated_dealers/{transaction_uid}_{dealer_uid}.png",
    log:
        "logs/event_study_plot_{statistic}_{transaction_uid}_{dealer_uid}.log",
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
    transaction_uids = pd.read_csv(
        checkpoints.nonempty_treatment_effects.get().output[0]
    )
    with open(checkpoints.statistics.get().output[0], "r") as f:
        statistics = f.read().splitlines()
    return [
        f"plots/event_study/{statistic}/treated_dealers/{x.transaction_uid}_{x.dealer_uid}.png"
        for x in transaction_uids.itertuples()
        for statistic in statistics
    ]


rule event_study_plots:
    input:
        "data/statistics.txt",
        "data/nonempty_treatment_effects.csv",
        get_event_study_plots,


rule event_study_cut_plot:
    input:
        "scripts/event_study_plot.jl",
        treatment_effects=get_nonempty_treatment_effects,
        dealer_demographics="data/dealer_demographics.arrow",
    output:
        "plots/event_study/{statistic}/cuts/{cut}.png",
    log:
        "logs/event_study_cut_plot_{statistic}_{cut}.log",
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
        "plots/event_study/{statistic}/cuts/{cut}.png",
        statistic=statistics,
        cut=[
            "grand_mean",
            "share_urban",
            "share_black",
            "share_hispanic",
            "median_income",
        ],
    )


rule event_study_cut_plots:
    input:
        "data/statistics.txt",
        get_event_study_cuts,


rule dealer_group_size_histogram:
    input:
        "scripts/dealer_group_size_histogram.jl",
        dealers_init="data/dealers_init.jld2",
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
        # run the script with the temp file as input
        shell(f"julia --project=. --threads=auto {scriptfile} < '{f.name}'")
        # delete the temp file
        os.remove(f.name)
