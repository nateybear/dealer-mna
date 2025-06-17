import ..Dealers: timetravel, State, add_distance
using ..Geometry: distance
import ..Registrations: registrations as all_registrations

using DataStructures: DefaultDict

# if group_remap is not provided, calc a straight market share
# if group_remap is provided, calculate change in market share
function market_share(
    registrations,
    period,
    group_remap = Dict();
    weight = LinearWeight(100mi),
)
    # ensure registrations are in the right period
    our_registrations = subset(registrations, :regeffdate => ByRow(in(period)))
    focal_uid = first(our_registrations.dealer_uid)
    focal_geometry = State.df.geometry[findfirst(==(focal_uid), State.df.uid)]

    # calc market share within each body class, then weight by
    # how important that body class is to the focal dealer
    body_class_weights = @> our_registrations begin
        groupby(:body_class)
        combine(nrow)
        transform(:nrow => (x -> x ./ sum(x)) => :class_weight)
    end

    body_classes = body_class_weights.body_class

    our_weights = DefaultDict{String, Float64}(0.0)
    for row in eachrow(body_class_weights)
        our_weights[row.body_class] = row.nrow
    end

    their_weights = DefaultDict{String, Float64}(0.0)

    # these are the guys that are changing hands
    contested_weights = DefaultDict{String, Float64}(0.0)

    # compile our group's and other groups' registrations
    timetravel(first(period)) do
        our_group_id = State.df.gid[findfirst(==(focal_uid), State.df.uid)]
        for dealer in eachrow(State.df)
            d = distance(dealer.geometry, focal_geometry)
            if dealer.uid == focal_uid || d > weight.max_distance
                continue
            end

            # get registrations for this dealer in the same period
            dealer_registrations = @> all_registrations() begin
                subset(
                    :regeffdate => ByRow(in(period)),
                    :dlrgdn => ByRow(in(dealer.gdns)),
                    :body_class => ByRow(in(body_classes)),
                )
                add_distance
            end

            if ismissing(dealer_registrations) || isempty(dealer_registrations)
                continue
            end

            subset!(dealer_registrations, :dealer_uid => ByRow(==(dealer.uid)))

            dealer_registrations = @> dealer_registrations begin
                groupby(:body_class)
                combine(nrow)
            end

            w = weight(d)

            dict = if haskey(group_remap, dealer.uid)
                contested_weights
            elseif dealer.gid == our_group_id
                our_weights
            else
                their_weights
            end

            for row in eachrow(dealer_registrations)
                dict[row.body_class] += row.nrow * w
            end
        end
    end

    function class_share(body_class, include_contested = false)
        class_weight = body_class_weights.class_weight[findfirst(
            ==(body_class), body_class_weights.body_class
        )]
        numerator =
            our_weights[body_class] + include_contested * contested_weights[body_class]
        denominator =
            our_weights[body_class] + their_weights[body_class] +
            contested_weights[body_class]

        return class_weight * numerator / denominator
    end

    return if isempty(contested_weights)
        sum(class_share(body_class) for body_class in body_classes)
    else
        sum(class_share(body_class, true) for body_class in body_classes) -
        sum(class_share(body_class, false) for body_class in body_classes)
    end
end
