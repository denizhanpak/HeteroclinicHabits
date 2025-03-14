using DifferentialEquations
using Plots
using Random
using StatsBase
using NLsolve  # Use NonlinearSolve instead of NLsolve
using LinearAlgebra  # Import norm function
using PlotlyJS  # Import PlotlyJS for interactive plots

name = "oscillator_network"

function approx_less_than(a, b, tol=1e-6)
    return a < (b + tol)
end



#ws = 1.0
#wi = -0.68
#we = 0.23
#delta = 0.4
#epsilon = delta / 8

params = (1.0, -0.68, 0.23, 0.4, 0.05)

function sigmoid(x, delta, epsilon)
    return 1 / (1 + exp(-(x - delta) / epsilon))
end

function sigmoid_prime(x, delta, epsilon)
    sig = sigmoid(x, delta, epsilon)
    return (exp(-(x - delta) / epsilon) / epsilon) * (sig^2)
end

function jacobian!(J, u, p)
    ws, wi, we, delta, epsilon = p
    sp = sigmoid_prime.(u, delta, epsilon)
    
    J[1, 1] = -1 + ws * sp[1]
    J[1, 2] = wi * sp[2]
    J[1, 3] = we * sp[3]

    J[2, 1] = we * sp[1]
    J[2, 2] = -1 + ws * sp[2]
    J[2, 3] = wi * sp[3]

    J[3, 1] = wi * sp[1]
    J[3, 2] = we * sp[2]
    J[3, 3] = -1 + ws * sp[3]
end

function vector_field!(du, u, p, t)
    ws, wi, we, delta, epsilon = p
    du[1] = -u[1] + ws * sigmoid(u[1], delta, epsilon) + wi * sigmoid(u[2], delta, epsilon) + we * sigmoid(u[3], delta, epsilon)
    du[2] = -u[2] + ws * sigmoid(u[2], delta, epsilon) + wi * sigmoid(u[3], delta, epsilon) + we * sigmoid(u[1], delta, epsilon)
    du[3] = -u[3] + ws * sigmoid(u[3], delta, epsilon) + wi * sigmoid(u[1], delta, epsilon) + we * sigmoid(u[2], delta, epsilon)
end

function find_roots(u0, params)
    function f!(F, x)
        vector_field!(F, x, params, 0.0)
    end
    function j!(J, x)
        jacobian!(J, x, params)
    end
    result = nlsolve(f!, j!, u0)
    result = nlsolve(f!, u0)
    return result.zero
end

function classify_stability(root, params)
    J = zeros(3, 3)
    jacobian!(J, root, params)
    eigenvalues, eigenvectors = eigen(J)

    if any(abs.(real.(eigenvalues)) .< 1e-7)
        return :center, eigenvalues
    elseif all(real.(eigenvalues) .< 0)
        return :stable, eigenvalues
    elseif all(real.(eigenvalues) .> 0)
        return :unstable, eigenvalues
    #elseif det(real.(eigenvectors)) ≈ 0
    #    return :degenerate, eigenvalues
    else
        return :saddle, eigenvalues
    end
end

function remove_duplicate_roots(roots, threshold=1e-6)
    unique_roots = []
    for root in roots
        if all(norm(root .- r) >= threshold for r in unique_roots)
            push!(unique_roots, root)
        end
    end
    return unique_roots
end

function grid_search_roots(grid_points, params, threshold=1e-6)
    roots = []
    for x in grid_points
        for y in grid_points
            for z in grid_points
                u0 = [x, y, z]
                root = find_roots(u0, params)
                if all(isfinite, root)
                    push!(roots, root)
                end
            end
        end
    end
    unique_roots = remove_duplicate_roots(roots, threshold)
    return unique_roots
end

function plot_saddles(root, params)
    J = zeros(3, 3)
    jacobian!(J, root, params)
    eigenvalues, eigenvectors = eigen(J)
    offset = 0.005
    trajectories = []
    for i in 1:length(eigenvalues)
        eigenvalue = real.(eigenvalues[i])
        eigenvector = real.(eigenvectors[:, i])
        time = 20
        u0 = root .+ offset * abs.(eigenvector)
        # Add a point at u0 for debugging
        #push!(trajectories, PlotlyJS.scatter3d(x=[u0[1]], y=[u0[2]], z=[u0[3]], mode="markers", marker=attr(size=5, color="black"), name="u0"))
        if real(eigenvalue) > 0
            prob = ODEProblem(vector_field!, u0, (0.0, time), params)
            sol = solve(prob, Tsit5(), saveat=0.001)  # Increased sampling resolution
            push!(trajectories, PlotlyJS.scatter3d(x=sol[1, :], y=sol[2, :], z=sol[3, :], mode="lines", line=attr(color="red", width=0.6), name="unstable"))
        elseif real(eigenvalue) < 0
            prob = ODEProblem(vector_field!, u0, (0.0, -time), params)
            sol = solve(prob, Tsit5(), saveat=0.001)  # Increased sampling resolution
            push!(trajectories, PlotlyJS.scatter3d(x=sol[1, :], y=sol[2, :], z=sol[3, :], mode="lines", line=attr(color="blue", width=0.6), name="stable"))
        end
    end
    return trajectories
end

function plot_phase_portrait(roots, solutions, limit_cycles, name, params)
    colors = []
    hovertexts = []
    trajectories = []
    for root in roots
        stability, eigenvalues = classify_stability(root, params)
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
            #append!(trajectories, plot_saddles(root, params))
        end
        push!(hovertexts, "Eigenvalues: " * string(eigenvalues))
    end
    scatter_data = PlotlyJS.scatter3d(x=[root[1] for root in roots], y=[root[2] for root in roots], z=[root[3] for root in roots], mode="markers", marker=attr(size=5, color=colors), text=hovertexts, hoverinfo="text")
    for (i, sol) in enumerate(solutions)
        push!(trajectories, PlotlyJS.scatter3d(x=sol[1, :], y=sol[2, :], z=sol[3, :], mode="lines", line=attr(width=0.4), name="trajectory $i"))
    end
    for (i, cycle) in enumerate(limit_cycles)
        push!(trajectories, PlotlyJS.scatter3d(x=cycle[1, :], y=cycle[2, :], z=cycle[3, :], mode="lines", line=attr(color="cyan", width=4), name="limit cycle $i"))
    end
    layout = Layout(
        title="Phase Portrait",
        scene=attr(
            xaxis=attr(title="Forward", range=[-1.1, 1.1]),
            yaxis=attr(title="Reverse", range=[-1.1, 1.1]),
            zaxis=attr(title="Turn", range=[-1.1, 1.1])
        )
    )
    plot = Plot([scatter_data; trajectories...], layout)
    PlotlyJS.savefig(plot, "$(name)_phase_portrait.html")
    display(plot)
end

function run_simulation(name, initial_conditions, tspan, params)
    solutions = []
    for u0 in initial_conditions
        prob = ODEProblem(vector_field!, u0, tspan, params)
        sol = solve(prob, Tsit5(), saveat=0.001)
        push!(solutions, sol)
    end
    return solutions
end

function find_limit_cycle(f, x, t, tol=5e-3)
    println("Finding limit cycle for initial condition: ", x)
    prob = ODEProblem(f, x, (0.0, t), params)
    sol = solve(prob, Tsit5(), saveat=0.01)
    
    # Find the index where the trajectory returns to the starting point
    start_index = findfirst(i -> norm(sol[i] .- x) < tol, 20:length(sol))
    
    if start_index === nothing
        println("No limit cycle found")
        return ([], 0.0)  # No limit cycle found
    end
    
    # Extract the limit cycle and the time it took to complete
    limit_cycle = sol[1:start_index]
    cycle_time = sol.t[start_index]
    
    return (limit_cycle, cycle_time)
end

function plot_limit_cycle(cycle, name)
    Plots.plot(title="Limit Cycle", xlabel="X", ylabel="Y", zlabel="Z", seriestype=:scatter, markersize=2)
    Plots.plot!(cycle[1, :], cycle[2, :], cycle[3, :], label="limit cycle", color="cyan", linewidth=4)
    Plots.savefig("$(name)_limit_cycle.png")
end

function make_hypersphere(r=0.5, d=3, n=10,o=nothing)
    if o == nothing
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

function filter_solution(sol, x_index, y_index, x_threshold=0.8, y_threshold=0.9)
    indices = findall(sol[x_index, :] .> x_threshold)
    if !isempty(indices)
        start_idx = indices[1]
        end_idx = findfirst(sol[y_index, start_idx:end] .> y_threshold)
        if end_idx !== nothing
            end_idx += start_idx - 1
            return sol[:, start_idx:end_idx]
        else
            return sol[:, start_idx:end]
        end
    end
    return nothing
end

function plot_oscillator_acceleration(path, dt)
    path = path[1:3, :]
    velocities = diff(path, dims=2) ./ dt
    accelerations = diff(velocities, dims=2) ./ dt
    Plots.plot(title="Oscillator Acceleration", xlabel="X", ylabel="Y", zlabel="Z", seriestype=:scatter, markersize=2, legend=false)
    Plots.plot!(accelerations[1, :], accelerations[2, :], accelerations[3, :], linewidth=1.2)
    Plots.savefig("oscillator_acceleration.png")
end

function plot_oscillator_force_field(path, dt, quiver_frequency=20)
    path = path[1:2, :]
    velocities = diff(path, dims=2) ./ dt
    accelerations = diff(velocities, dims=2) ./ dt

    orthogonal_path = []
    for point in eachcol(path)
        orthogonal_point = [point[2], -point[1]]
        push!(orthogonal_path, orthogonal_point)
    end

    forces = []
    for point in zip(eachcol(accelerations), orthogonal_path)
        force_point = dot(point[1], point[2])
        push!(forces, force_point)
    end
    max = maximum(abs.(forces))
    forces_normalized = (forces)./max * 0.05

    # Plot horizontal line
    x_values = 0:dt:(dt * (length(forces)))
    y_values = zeros(length(x_values)) .+ 0.05
    Plots.plot(x_values, zeros(length(x_values)), color="black", linewidth=1.2)

    # Plot quiver every quiver_frequency points
    quiver_indices = 1:quiver_frequency:length(x_values)
    for i in quiver_indices
        mag = forces_normalized[i] + 0.01
        quiver!([x_values[i]], [y_values[i]], quiver=mag, linewidth=0.3, color="red")
        quiver!([x_values[i]], -[y_values[i]], quiver=-mag, linewidth=0.3, color="red")
    end

    Plots.title!("Oscillator Force Field")
    Plots.xlabel!("Time")
    Plots.ylabel!("Force")
    Plots.savefig("oscillator_force_field.png")
end

function plot_oscillator_channel(source, sink, params, r=0.2, n=20, dt=0.001)
    max_dim_source = argmax(abs.(source))
    max_dim_sink = argmax(abs.(sink))
    
    y_index = max_dim_source
    x_index = max_dim_sink
    
    Plots.plot(title="Oscillator Channel", xlabel="Dimension $x_index", ylabel="Dimension $y_index", seriestype=:scatter, markersize=2, xlims=(-0.1, 1.1), ylims=(-0.1, 1.1), legend=false)

    initial_conditions = make_hypersphere(r, 3, n, [1,0.01,0.01])
    tspan = (0.0, 20.0)
    
    for u0 in initial_conditions
        prob = ODEProblem(vector_field!, u0, tspan, params)
        sol = solve(prob, Tsit5(), reltol=1e-7, abstol=1e-7, saveat=dt)
        Plots.plot!(sol[y_index, :], sol[x_index, :], linewidth=1.2)
        
        #filtered_sol = filter_solution(sol, x_index, y_index, 0.2, 0.2)
        #if filtered_sol !== nothing
        #    Plots.plot!(filtered_sol[y_index, :], filtered_sol[x_index, :], linewidth=1.2)
        #end
    end
    
    Plots.scatter!([source[x_index]], [source[y_index]], color="red", markersize=5)
    Plots.scatter!([sink[x_index]], [sink[y_index]], color="blue", markersize=5)
    
    # Add a thick black line based on the specific initial condition [0.999, 0.0001, 0.0001]
    line_ic = [0.9999, 0.001, 0.0001]
    line_prob = ODEProblem(vector_field!, line_ic, 7.55, params)
    line_sol = solve(line_prob, Tsit5(), reltol=1e-7, abstol=1e-7, saveat=dt)
    
    Plots.plot!(line_sol[y_index, :], line_sol[x_index, :], linewidth=10, arrow=:arrow, color="black", la=0.2)
    filtered_line_sol = filter_solution(line_sol, x_index, y_index, .01, .95)
    if filtered_line_sol !== nothing
        Plots.plot!(filtered_line_sol[y_index, :], filtered_line_sol[x_index, :], linewidth=10, arrow=:arrow, color="black", la=0.2)
    end
    #println(size(line_sol))
    #Plots.plot!(line_sol[y_index, :], line_sol[x_index, :], linewidth=10, arrow=:arrow, color="black", la=0.2)
    
    Plots.savefig("oscillator_channel.png")
    return line_sol
end

line = plot_oscillator_channel([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], params)
plot_oscillator_force_field(line, 0.001, 250)
exit()

# Run the simulation and generate plots with more initial conditions
initial_conditions = [[-0.5290695,0.6016491, 0.8298354],[-0.6454127, 0.979755, 0.2632572]]
initial_conditions = make_hypersphere(1, 3, 10, [0.5, 0.5, 0.5])
tspan = (0.0, 500.0)
solutions = run_simulation(name, initial_conditions, tspan, params)
#solutions = []



# Generate roots from a grid search
grid_points = range(-.8, stop=.8, length=10)
#roots = grid_search_roots(grid_points, params)

# Find limit cycles
limit_cycles = []
limit_cycle_ics = [[-0.6630017, 0.9702522, 0.2693012]]
for u0 in limit_cycle_ics
    cycle, _ = find_limit_cycle(vector_field!, u0, 30.0)
    if !isempty(cycle)
        push!(limit_cycles, cycle)
    end
end

# Plot limit cycle separately
if !isempty(limit_cycles)
    plot_limit_cycle(limit_cycles[1], name)
end

# Plot phase portrait with trajectories and limit cycles
#plot_phase_portrait(roots, solutions, limit_cycles, name, params)
p = plot_phase_portrait(roots, [], [], name, params)
