using DealerMnA

using DataFrames
import JLD2
using UUIDs
using Unitful
using PartialFunctions

!isempty(ARGS) && snakemake!(ARGS[1])

with_logging() do
    Dealers.load_from_disk!(snakemake.input["dealers_init"])
    dealer_uid = UUID(snakemake.wildcards["dealer_uid"])
    transaction_uid = UUID(snakemake.wildcards["transaction_uid"])
    transactions = JLD2.load_object(snakemake.input["transactions"])
    t = first(t for t in transactions if t.uid == transaction_uid)
    dealer = pluck(Dealers.State.df, :uid => ByRow(==(dealer_uid)))
    @info "Dealer under consideration" dealer
    intersects = !isempty ∘ intersect
    franchise = snakemake.wildcards["dealer_type"] == "franchise"
    relevant_dealers = subset(
        Dealers.State.df,
        (franchise ? [:franchises => ByRow(intersects$dealer.franchises)] : [])...,
        :uid => ByRow(!=(dealer_uid)),
    )
    distances = Geometry.distance.(relevant_dealers.geometry, Ref(dealer.geometry))
    order = sortperm(distances)
    open(snakemake.output[1], "w") do io
        count = 0
        for i in order
            if count == 3
                break
            end
            dealer_uid = relevant_dealers.uid[i]
            treated_registrations = Empirics.treated_registrations(t; dealer_uid)
            if !ismissing(treated_registrations) &&
                !isempty(treated_registrations) &&
                treated_registrations.τ ⊇ Empirics.VALID_PERIODS
                count += 1
                @info "Writing $(count)th closest dealership $(relevant_dealers.uid[i])" distance = distances[i]
                println(io, dealer_uid)
            end
        end
    end
end
