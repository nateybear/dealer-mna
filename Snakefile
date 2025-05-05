import glob

julia_src = glob.glob("src/**/*.jl")
julia_dealers = glob.glob("src/dealers/**/*.jl")
julia_empirics = glob.glob("src/empirics/**/*.jl")


def transaction_uids(fname):
    """
    Get the transaction uids from the treated_registrations folder.
    """
    if exists(fname):
        with open(fname) as f:
            return f.read().splitlines()


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


rule transaction_uids:
    input:
        *julia_src,
        "scripts/transaction_uids.jl",
        transactions_for_analysis="data/transactions_for_analysis.jld2",
    output:
        "data/transaction_uids.txt",
    log:
        "logs/transaction_uids.log",
    run:
        julia("scripts/transaction_uids.jl", input=input, output=output, log=log)


rule treated_registrations:
    input:
        *julia_src,
        "scripts/treated_registrations.jl",
        transactions_for_analysis="data/transactions_for_analysis.jld2",
        dealers_init="data/dealers_init.jld2",
    output:
        expand(
            [
                "data/treated_registrations/last_registrations_before_{uid}.arrow",
                "data/treated_registrations/first_registrations_after_{uid}.arrow",
            ],
            uid=transaction_uids(rules.transaction_uids.output[0]),
        ),
    log:
        "logs/treated_registrations.log",
    run:
        julia(
            "scripts/treated_registrations.jl",
            input=input,
            output=output,
            log=log,
        )


rule control_registrations:
    input:
        *julia_src,
        "scripts/control_registrations.jl",
        transactions_for_analysis="data/transactions_for_analysis.jld2",
        dealers_init="data/dealers_init.jld2",
    output:
        expand(
            [
                "data/control_registrations/last_registrations_before_{uid}.arrow",
                "data/control_registrations/first_registrations_after_{uid}.arrow",
            ],
            uid=transaction_uids(rules.transaction_uids.output[0]),
        ),
    log:
        "logs/control_registrations.log",
    run:
        julia(
            "scripts/control_registrations.jl",
            input=input,
            output=output,
            log=log,
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
