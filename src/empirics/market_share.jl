import ..Dealers: timetravel, fuzzytravel, State, add_distance
using ..Geometry: distance
import ..Registrations: registrations
import ..pluck

using DataStructures: DefaultDict
using Base.Threads # let's speed this up
import LinearAlgebra: BLAS

# if group_remap is not provided, calc a straight market share
# if group_remap is provided, calculate change in market share
function market_share(
    focal_uid,
    period,
    group_remap = Dict();
    weight = LinearWeight(50mi),
)
    focal_dealer = fuzzytravel(minimum(period)) do
        NamedTuple(pluck(State.df, :uid => ByRow(==(focal_uid))))
    end
    # cases: empty group is just compute a market share in the period
    # group_remap contains focal_uid: compare share of current gid to new gid
    # group_remap does not: compare share of focal gid before and after remap
    if isempty(group_remap)
        our_sales = their_sales = 0
        _market_shares(focal_dealer, period, weight) do dealer, sales
            if dealer.gid == focal_dealer.gid
                our_sales += sales
            else
                their_sales += sales
            end
        end
        return our_sales / (our_sales + their_sales)
    elseif collect(keys(group_remap)) == [focal_dealer.uid]
        new_gid = only(values(group_remap))
        my_sales = old_sales = new_sales = other_sales = 0
        _market_shares(focal_dealer, period, weight) do dealer, sales
            if dealer.uid == focal_dealer.uid
                my_sales += sales
            elseif dealer.gid == focal_dealer.gid
                old_sales += sales
            elseif dealer.gid == new_gid
                new_sales += sales
            else
                other_sales += sales
            end
        end
        return (new_sales - old_sales) / (new_sales + old_sales + my_sales + other_sales)
    else
        im_acquiring = only(unique(values(group_remap))) == focal_dealer.gid
        transated_sales = other_sales = 0
        _market_shares(focal_dealer, period, weight) do dealer, sales
            if dealer.uid in keys(group_remap)
                transated_sales += sales
            else
                other_sales += sales
            end
        end
        return (transated_sales * im_acquiring - transated_sales * !im_acquiring) /
               (transated_sales + other_sales)
    end
end

function hhi(
    focal_uid,
    period,
    group_remap = Dict();
    weight = LinearWeight(50mi),
)
    focal_dealer = fuzzytravel(minimum(period)) do
        NamedTuple(pluck(State.df, :uid => ByRow(==(focal_uid))))
    end
    _hhi(group_sales) =
        let shares = values(group_sales) ./ sum(values(group_sales))
            10_000sum(shares .^ 2)
        end
    if isempty(group_remap)
        group_sales = DefaultDict{UUID, Float64}(0.0)
        _market_shares(focal_dealer, period, weight) do dealer, sales
            group_sales[dealer.gid] += sales
        end
        return _hhi(group_sales)
    else
        transacted_sales = 0
        group_sales = DefaultDict{UUID, Float64}(0.0)
        _market_shares(focal_dealer, period, weight) do dealer, sales
            if dealer.uid in keys(group_remap)
                transacted_sales += sales
            else
                group_sales[dealer.gid] += sales
            end
        end

        before_sales = deepcopy(group_sales)
        before_gid = fuzzytravel(minimum(period)) do
            let transacted_dealers = subset(
                    State.df, :uid => ByRow(in(keys(group_remap))), view = true
                )
                only(unique(transacted_dealers.gid))
            end
        end
        before_sales[before_gid] += transacted_sales

        after_sales = deepcopy(group_sales)
        after_gid = only(unique(values(group_remap)))
        after_sales[after_gid] += transacted_sales

        return _hhi(after_sales) - _hhi(before_sales)
    end
end

# hard code this for missing data
const OVERALL_SHARE = DataFrame(
    body_class = ["SUV/Crossover", "Sedan", "Coupe", "Pickup", "Van"],
    class_weight = [
        0.42604284545328114,
        0.23097072302825422,
        0.027261662148870575,
        0.2982546800785856,
        0.01747008929100848,
    ],
)

function _market_shares(
    handle,
    focal_dealer,
    period,
    weight,
)
    our_registrations = fuzzytravel(minimum(period)) do
        @> registrations() begin
            subset(
                :t => ByRow(in(period)),
                :dlrgdn => ByRow(in(focal_dealer.gdns)),
                view = true,
            )
        end
    end

    # calc market share within each body class, then weight by
    # how important that body class is to the focal dealer
    body_class_weights = if isempty(our_registrations)
        OVERALL_SHARE
    else
        @> our_registrations begin
            groupby(:body_class)
            combine(nrow)
            transform(:nrow => (x -> x ./ sum(x)) => :class_weight)
            select(Not(:nrow))
        end
    end

    body_classes = body_class_weights.body_class

    # compile our group's and other groups' registrations
    fuzzytravel(maximum(period)) do
        nearby = subset(
            State.df,
            :geometry =>
                ByRow(x -> distance(x, focal_dealer.geometry) < max_distance(weight)),
            :active => identity,
            view = true,
        )

        nearby_registrations = @> registrations() begin
            subset(
                :t => ByRow(in(period)),
                :body_class => ByRow(in(body_classes)),
                view = true,
            )
            select(:dlrgdn, :body_class)
        end

        weighted_sales = @> nearby begin
            transform(
                :geometry => ByRow(x -> distance(x, focal_dealer.geometry)) => :distance
            )
            transform(:distance => ByRow(d -> weight(d)) => :distance_weight)
            flatten(:gdns)
            unique([:gdns, :gid])
            innerjoin(nearby_registrations, on = :gdns => :dlrgdn)
            groupby([:uid, :body_class])
            combine(nrow, :distance_weight => only ∘ unique => :distance_weight)
            innerjoin(body_class_weights, on = :body_class)
            groupby(:uid)
            combine(
                [:distance_weight, :nrow, :class_weight] =>
                    ((x, y, z) -> first(x) * sum(y .* z)) => :weighted_sales,
            )
            innerjoin(State.df, on = :uid)
        end

        for r in eachrow(weighted_sales)
            handle(r, r.weighted_sales)
        end
    end
end
