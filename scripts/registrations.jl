using DealerMnA

using Random
Random.seed!(1234)

# use a set of hashes to filter out duplicates. Low chance of collision
new_hashset() = Set{UInt64}()
function makeunique(df, hashset)
    hashes = hash.(eachrow(df))
    mask = map(hashes) do h
        if h in hashset
            return false
        else
            push!(hashset, h)
            return true
        end
    end
    return df[mask, :]
end

!isempty(ARGS) && snakemake!(ARGS[1])

with_logging() do
    rm(snakemake.output["registrations"], force = true)
    open(Arrow.Writer, snakemake.output["registrations"]) do out_writer
        hashset = new_hashset()

        # "implicit" filtering since vins are only decoded for registrations with odometer < 250
        vins = select(
            DataFrame(Arrow.Table(snakemake.input["vins_decoded"])),
            Not(r"displacement"),
        )
        for fname in shuffle(snakemake.input["registration_files"])
            @info "Processing registration file: $fname"
            for registrations in Registrations.load_registration_file_chunked(fname)
                registrations = makeunique(registrations, hashset)
                registrations = innerjoin(registrations, vins, on = :vin)
                Arrow.write(out_writer, registrations)
            end
        end
    end
end
