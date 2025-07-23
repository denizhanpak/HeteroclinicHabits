include("./DwellTime.jl")

struct DS
    dimensions::Int
    vector_field::Function
    jacobian::Function
    noise::Function
    plot_transform::Function
    parameters::Tuple
end

function find_roots(u0::Vector, ds::DS)
    if length(u0) != ds.dimensions
        throw(ArgumentError("Initial condition must have the same length as the number of dimensions"))
    end

    function f!(F, x)
        ds.vector_field(F, x, ds.parameters, 0.0)
    end
    function j!(J, x)
        ds.jacobian(J, x, ds.parameters)
    end
    result = nlsolve(f!, j!, u0)
    
    return result.zero
end


function classify_stability(solution::Vector, ds::DS)
    J = ds.jacobian(solution, ds.parameters)
  
    eigenvalues = eigvals(J)
    println(eigenvalues)
    if any(abs.(real.(eigenvalues)) .< 1e-6)
        return :center, eigenvalues
    elseif all(real.(eigenvalues) .< 0)
        return :stable, eigenvalues
    elseif all(real.(eigenvalues) .> 0)
        return :unstable, eigenvalues
    else
        return :saddle, eigenvalues
    end
end

function remove_duplicate_roots(roots::Vector, threshold::Real=1e-6)
    unique_roots = []
    for root in roots
        if all(norm(root .- r) >= threshold for r in unique_roots)
            push!(unique_roots, root)
        end
    end
    return unique_roots
end

function grid_search_roots(ds::DS, threshold::Real=1e-6, mesh_range::Tuple=(0,1.1), mesh_size::Int=10)
    grid_points = collect(range(mesh_range[1], mesh_range[2], length=mesh_size))
    roots = []
    for x in grid_points
        for y in grid_points
            for z in grid_points
                u0 = [x, y, z]
                root = find_roots(u0, ds)
                if all(isfinite, root)
                    push!(roots, root)
                end
            end
        end
    end
    unique_roots = remove_duplicate_roots(roots, threshold)
    return unique_roots
end

function run_simulation(ds::DS, initial_conditions::Vector, tspan::Tuple=(0.0, 50.0))
    solutions = []
    if ds.noise === nothing
        for u0 in initial_conditions
            prob = ODEProblem(ds.vector_field, u0, tspan, ds.parameters)
            sol = solve(prob, Tsit5(), saveat=0.01)
            push!(solutions, sol)
        end
    else
        for u0 in initial_conditions
            prob = SDEProblem(ds.vector_field, ds.noise, u0, tspan, ds.parameters)
            sol = solve(prob, saveat=0.01)
            push!(solutions, sol)
        end
    end
    return solutions
end

function plot_time_series(ds::DS, solution_ICs::Vector, tspan::Tuple=(0.0,50.0), name::String="time_series")
    solutions = run_simulation(ds, solution_ICs, tspan)
    plot = Plots.plot(title=name, xlabel="Time", ylabel="Value", zlabel="Dimension", seriestype=:scatter, markersize=2)
    for (i, sol) in enumerate(solutions)
        time = sol.t
        for j in 1:ds.dimensions
            Plots.plot!(time, ds.plot_transform.(sol[j, :]), label="$i dim_$j", linewidth=0.7)
        end
    end
    return plot, solutions
    #Plots.savefig("$(name)_time_series.png")
end 

function plot_phase_portrait(ds::DS, solution_ICs::Vector, root_range::Tuple=(0.1,1.1), mesh_size::Int=10, limit_cycle_ics::Vector=[], name::String="phase_portrait")
    #plot the roots
    roots = grid_search_roots(ds, 1e-6, root_range, mesh_size)
    colors = []
    hovertexts = []
    trajectories = []
    for root in roots
        stability, eigenvalues = classify_stability(root, ds)
        if stability == :stable
            push!(colors, "blue")
            #append!(trajectories, plot_saddles(root, params))
        elseif stability == :unstable
            push!(colors, "red")
            #append!(trajectories, plot_saddles(root, params))
        elseif stability == :center
            push!(colors, "yellow")
        elseif stability == :degenerate
            push!(colors, "purple")
        else
            push!(colors, "green")
            append!(trajectories, plot_saddles(root, params))
        end
        push!(hovertexts, "Eigenvalues: " * string(eigenvalues))
    end
    scatter_data = PlotlyJS.scatter3d(x=[root[1] for root in roots], y=[root[2] for root in roots], z=[root[3] for root in roots], mode="markers", marker=attr(size=5, color=colors), text=hovertexts, hoverinfo="text")
    
    #plot the trajectories
    solutions = run_simulation(ds, solution_ICs)
    for (i, sol) in enumerate(solutions)
        push!(trajectories, PlotlyJS.scatter3d(x=sol[1, :], y=sol[2, :], z=sol[3, :], mode="lines", line=attr(width=0.4), name="trajectory $i"))
    end

    #plot the limit cycles
    limit_cycles = run_simulation(ds, limit_cycle_ics)
    for (i, cycle) in enumerate(limit_cycles)
        push!(trajectories, PlotlyJS.scatter3d(x=cycle[1, :], y=cycle[2, :], z=cycle[3, :], mode="lines", line=attr(color="cyan", width=4), name="limit cycle $i"))
    end
    layout = Layout(
        title="Phase Portrait",
        scene=attr(
            xaxis=attr(title="Forward", range=[-0.1, 1.1]),
            yaxis=attr(title="Reverse", range=[-0.1, 1.1]),
            zaxis=attr(title="Turn", range=[-0.1, 1.1])
        )
    )
    return Plot([scatter_data; trajectories...], layout)
    #PlotlyJS.savefig(plot, "$(name)_phase_portrait.html")
    #display(plot)
end


function make_hypersphere(r=0.5, d=3, n=10,o=nothing)
    if o === nothing
        o = zeros(d)
    end
    if length(o) != d
        error("The origin must have the same dimension as the hypersphere")
        return
    end
    points = []
    for _ in 1:n
        point = randn(d)
        point /= norm(point)
        point *= r * rand()
        point += o
        push!(points, point)
    end
    return points
end

function solve_SDE(ic::Vector, ds::DS, tspan::Tuple=(0.0, 50.0))
    prob = SDEProblem(ds.vector_field, ds.noise, ic, tspan, ds.parameters)
    sol = solve(prob, saveat=0.01)
    return sol
end

function solve_ODE(ic::Vector, ds::DS, tspan::Tuple=(0.0, 50.0))
    prob = ODEProblem(ds.vector_field, ic, tspan, ds.parameters)
    sol = solve(prob, Tsit5(), saveat=0.01)
    return sol
end

function plot_saddles(point::Vector, ds::DS)
    J = zeros(ds.dimensions, ds.dimensions)
    jac = ds.jacobian(J, point, ds.parameters)
    eigenvalues, eigenvectors = eigen(jac)
    offset = 0.005
    trajectories = []
    for i in eachindex(eigenvalues)
        eigenvalue = real.(eigenvalues[i])
        eigenvector = real.(eigenvectors[:, i])
        time = 50
        u0 = root .+ offset * abs.(eigenvector)
        # Add a point at u0 for debugging
        #push!(trajectories, PlotlyJS.scatter3d(x=[u0[1]], y=[u0[2]], z=[u0[3]], mode="markers", marker=attr(size=5, color="black"), name="u0"))
        if real(eigenvalue) > 0
            sol = solve_ODE(u0, ds, (0.0, time))
            push!(trajectories, PlotlyJS.scatter3d(x=sol[1, :], y=sol[2, :], z=sol[3, :], mode="lines", line=attr(color="red", width=0.6), name="unstable"))
        elseif real(eigenvalue) < 0
            sol = solve_ODE(u0, ds, (0.0, time))
            push!(trajectories, PlotlyJS.scatter3d(x=sol[1, :], y=sol[2, :], z=sol[3, :], mode="lines", line=attr(color="red", width=0.6), name="unstable"))
        end
    end
    return trajectories
end

function evaluate_dwell_times(ds::DS, sample_time::Tuple=(0,20000), ic_count::Int=10)
    ics = make_hypersphere(0.1, ds.dimensions, ic_count, [0.53, 0.53, 0.53, 0.53])
    solutions = run_simulation(ds, ics, sample_time)
    bins = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    dwell_times = []
    for sol in solutions
        adt = average_dwell_times(sol, bins)
        rv=[adt[i] / minimum(values(adt)) for i in [1, 2, 3, 4]]
        push!(dwell_times, rv)
    end

    return hcat(dwell_times...)
end

function evaluate_sequence_generation(ds::DS, sequence::Vector{Int}, sample_time::Tuple, initial_conditions::Vector; threshold::Real=0.98, O::Real=100.0, tmax::Real=1000.0, tmin::Real=0.0)
    # Simulate the dynamical system
    if ds.noise === nothing
        prob = ODEProblem(ds.vector_field, initial_conditions, sample_time, ds.parameters)
        sol = solve(prob, Tsit5(), saveat=0.01)
    else
        prob = SDEProblem(ds.vector_field, ds.noise, initial_conditions, sample_time, ds.parameters)
        sol = solve(prob, saveat=0.01)
    end
    
    # Find crossings for each dimension
    all_crossings = Dict{Int, Tuple{Vector{Int}, Vector{Int}}}()
    for dim in 1:ds.dimensions
        dimension_data = [sol.u[i][dim] for i in 1:length(sol.u)]
        up_crossings, down_crossings = find_crossings(dimension_data, threshold=threshold)
        all_crossings[dim] = (up_crossings, down_crossings)
    end
    
    # Check if the sequence is valid
    if length(sequence) > ds.dimensions
        throw(ArgumentError("Sequence length cannot exceed system dimensions"))
    end
    
    # Validate the sequence occurs in order
    sequence_valid = true
    sequence_times = []
    
    for i in 1:length(sequence)
        current_dim = sequence[i]
        up_crossings, down_crossings = all_crossings[current_dim]
        
        if isempty(up_crossings)
            sequence_valid = false
            break
        end
        
        # Find the first up crossing that occurs after the previous sequence element
        valid_crossing_found = false
        for up_time in up_crossings
            # Check if this crossing occurs after the previous sequence element
            if i == 1 || up_time > sequence_times[end][2] # After previous down crossing
                # Find corresponding down crossing
                down_candidates = filter(d -> d > up_time, down_crossings)
                if !isempty(down_candidates)
                    down_time = minimum(down_candidates)
                    
                    # If this is not the last element, check overlap with next dimension
                    if i < length(sequence)
                        next_dim = sequence[i+1]
                        next_up_crossings, _ = all_crossings[next_dim]
                        
                        # Check if next dimension has an up crossing within overlap O of current down crossing
                        overlap_valid = false
                        for next_up in next_up_crossings
                            if evaluate_crossings(up_time, down_time, next_up, tmax=tmax, tmin=tmin, O=O)
                                overlap_valid = true
                                break
                            end
                        end
                        
                        if overlap_valid
                            push!(sequence_times, (up_time, down_time))
                            valid_crossing_found = true
                            break
                        end
                    else
                        # Last element in sequence, no overlap check needed
                        push!(sequence_times, (up_time, down_time))
                        valid_crossing_found = true
                        break
                    end
                end
            end
        end
        
        if !valid_crossing_found
            sequence_valid = false
            break
        end
    end
    
    return sequence_valid, sequence_times, all_crossings
end

function evaluate_crossings(ui::Int, di::Int, uj::Int; tmax::Real=1000.0, tmin::Real=0.0, O::Real=100.0)
    t = di - ui
    if t > tmax || t < tmin
        return false
    end
    o = abs(uj - di)
    if o < O && o > 0
        return true
    end
    return false
end

function find_crossings(ts::Vector; threshold::Real=0.98)
    up_crossings = []
    down_crossings = []
    for (index, step1) in enumerate(ts[1:end-1])
        step2 = ts[index+1]
        if step1 <= threshold && step2 > threshold
            push!(up_crossings, index+1)
        elseif step1 >= threshold && step2 < threshold
            push!(down_crossings, index+1)
        end
    end
    return (up_crossings, down_crossings)
end

function find_limit_cycle(ds::DS, x::Vector, t::Real, tol::Real=5e-4)
    prob = ODEProblem(ds.vector_field, x, (0.0, t), ds.parameters)
    sol = solve(prob, Tsit5(), saveat=0.01)
    
    # Find the index where the trajectory returns to the starting point
    start_index = findfirst(i -> norm(sol[i] .- x) < tol, 20:length(sol))
    
    if start_index === nothing
        return ([], 0.0)  # No limit cycle found
    end
    
    # Extract the limit cycle and the time it took to complete
    limit_cycle = sol[1:start_index]
    cycle_time = sol.t[start_index]
    
    return (limit_cycle, cycle_time)
end
