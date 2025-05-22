using LinearAlgebra

function average_dwell_times(solution, bins)
    # Extract time points and states from the solution
    t = solution.t
    u = [x .* x for x in solution.u]
    n = length(t)
    
    # Handle cases with insufficient data
    avg_dwell = Dict{Int, Float64}()
    if n < 2
        for bin_idx in eachindex(bins)
            avg_dwell[bin_idx] = 0.0
        end
        return avg_dwell
    end

    # Assign each interval to the closest bin
    bin_assignments = Vector{Int}(undef, n-1)
    for i in 1:n-1
        distances = [norm(u[i] - bin) for bin in bins]
        _, closest_idx = findmin(distances)
        bin_assignments[i] = closest_idx
    end

    # Track dwell time periods for each bin
    visits = Dict{Int, Vector{Float64}}()
    current_bin = bin_assignments[1]
    current_duration = t[2] - t[1]

    for i in 2:n-1
        if bin_assignments[i] == current_bin
            current_duration += t[i+1] - t[i]
        else
            # Record the completed visit
            if haskey(visits, current_bin)
                push!(visits[current_bin], current_duration)
            else
                visits[current_bin] = [current_duration]
            end
            # Start new visit
            current_bin = bin_assignments[i]
            current_duration = t[i+1] - t[i]
        end
    end

    # Add the final visit
    if haskey(visits, current_bin)
        push!(visits[current_bin], current_duration)
    else
        visits[current_bin] = [current_duration]
    end

    # Calculate average dwell times for all bins
    for bin_idx in eachindex(bins)
        if haskey(visits, bin_idx)
            dwells = visits[bin_idx]
            avg_dwell[bin_idx] = sum(dwells) / length(dwells)
        else
            avg_dwell[bin_idx] = 0.0
        end
    end

    return avg_dwell
end
