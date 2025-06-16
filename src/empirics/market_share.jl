import ..Dealers: timetravel, State, add_distance
using ..Geometry: distance
import ..Registrations: registrations as all_registrations

function market_share(
    registrations,
    period,
    weight = LinearWeight(100mi),
)
    # ensure registrations are in the right period
    our_registrations = subset(registrations, :regeffdate => ByRow(in(period)))
    focal_uid = first(our_registrations.dealer_uid)
    focal_geometry = State.df.geometry[findfirst(==(focal_uid), State.df.uid)]
    our_registrations.weight .= 1.0
    body_classes = unique(our_registrations.body_class)
    body_class_counts = @> our_registrations groupby(:body_class) combine(nrow)

    other_registrations = DataFrame()

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

            # calculate distance to focal dealer
            dealer_registrations.distance .= d

            # apply weight based on distance
            dealer_registrations.weight .= weight(d)

            # append to our/other registrations
            append!(
                dealer.gid == our_group_id ? our_registrations : other_registrations,
                dealer_registrations,
                cols = :union,
            )
        end
    end

    our_body_class_sum = @> our_registrations begin
        groupby(:body_class)
        combine(:weight => sum => :our_weight)
    end
    other_body_class_sum = @> other_registrations begin
        groupby(:body_class)
        combine(:weight => sum => :other_weight)
    end

    out = @> body_class_counts begin
        innerjoin(our_body_class_sum, on = :body_class)
        innerjoin(other_body_class_sum, on = :body_class)
        transform(
            :nrow => (x -> x ./ sum(x)) => :class_weight,
            [:our_weight, :other_weight] => ByRow(/) => :market_share,
        )
    end

    return sum(out.market_share .* out.class_weight)
end
