import glob
import pandas as pd

julia_src = glob.glob("src/**/*.jl")
julia_dealers = glob.glob("src/dealers/**/*.jl")
julia_empirics = glob.glob("src/empirics/**/*.jl")


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
        "data/price_table.jld2",
    log:
        "logs/price_table.log",
    run:
        julia("scripts/price_table.jl", output=output, log=log)


rule transactions_for_analysis:
    input:
        *julia_dealers,
        *julia_empirics,
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
        "scripts/treated_registrations.jl",
        transactions_for_analysis="data/transactions_for_analysis.jld2",
        dealers_init="data/dealers_init.jld2",
        price_table="data/price_table.jld2",
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
        "scripts/control_registrations.jl",
        transactions_for_analysis="data/transactions_for_analysis.jld2",
        dealers_init="data/dealers_init.jld2",
        price_table="data/price_table.jld2",
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


rule treatment_effects_summary:
    input:
        get_treatment_effects,


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
