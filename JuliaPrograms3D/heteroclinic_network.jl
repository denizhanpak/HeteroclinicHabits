using DifferentialEquations
using Plots
using Random
using StatsBase
using NLsolve  # Use NonlinearSolve instead of NLsolve
using LinearAlgebra  # Import norm function
using PlotlyJS  # Import PlotlyJS for interactive plots

name = "heteroclinic_network"

#mu = 1
#a = 1.0
#b = 0.55
#c = 1.5
params = (1.0, 1.0, 1.1, 2)

function jacobian!(J, u, p)
    mu, a, b, c = p
    J[1, 1] = mu - 3 * a * u[1]^2 - b * u[2]^2 - c * u[3]^2
    J[1, 2] = -2 * b * u[1] * u[2]
    J[1, 3] = -2 * c * u[1] * u[3]

    J[2, 1] = -2 * c * u[2] * u[1]
    J[2, 2] = mu - 3 * a * u[2]^2 - b * u[3]^2 - c * u[1]^2
    J[2, 3] = -2 * b * u[2] * u[3]
    
    J[3, 1] = -2 * b * u[3] * u[1]
    J[3, 2] = -2 * c * u[3] * u[2]
    J[3, 3] = mu - 3 * a * u[3]^2 - b * u[1]^2 - c * u[2]^2
end

function vector_field!(du, u, p, t)
    mu, a, b, c = p
    du[1] = mu * u[1] - u[1] * (a * u[1]^2 + b * u[2]^2 + c * u[3]^2)
    du[2] = mu * u[2] - u[2] * (a * u[2]^2 + b * u[3]^2 + c * u[1]^2)
    du[3] = mu * u[3] - u[3] * (a * u[3]^2 + b * u[1]^2 + c * u[2]^2)
end

function neg_vector_field!(du, u, p, t)
    vector_field!(du, u, p, t)
    du .*= -1
end

function find_roots(u0, params)
    function f!(F, x)
        vector_field!(F, x, params, 0.0)
    end
    function j!(J, x)
        jacobian!(J, x, params)
    end
    result = nlsolve(f!, j!, u0)
    #result = nlsolve(f!,  u0)
    return result.zero
end

function classify_stability(root, params)
    J = zeros(3, 3)
    jacobian!(J, root, params)
    eigenvalues = eigvals(J)
    if all(real.(eigenvalues) .< 0)
        return :stable, eigenvalues
    elseif all(real.(eigenvalues) .> 0)
        return :unstable, eigenvalues
    else
        return :saddle, eigenvalues
    end
end

function remove_duplicate_roots(roots, threshold=1e-3)
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
        time = 50
        u0 = root .+ offset * abs.(eigenvector)
        # Add a point at u0 for debugging
        #push!(trajectories, PlotlyJS.scatter3d(x=[u0[1]], y=[u0[2]], z=[u0[3]], mode="markers", marker=attr(size=5, color="black"), name="u0"))
        if real(eigenvalue) > 0
            prob = ODEProblem(vector_field!, u0, (0.0, time), params)
            sol = solve(prob, Tsit5(), saveat=0.01)  # Increased sampling resolution
            push!(trajectories, PlotlyJS.scatter3d(x=sol[1, :], y=sol[2, :], z=sol[3, :], mode="lines", line=attr(color="red", width=0.6), name="unstable"))
        elseif real(eigenvalue) < 0
            prob = ODEProblem(vector_field!, u0, (0.0, -time), params)
            sol = solve(prob, Tsit5(), saveat=0.01)  # Increased sampling resolution
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
            append!(trajectories, plot_saddles(root, params))
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
            xaxis=attr(title="Forward", range=[-0.1, 1.1]),
            yaxis=attr(title="Reverse", range=[-0.1, 1.1]),
            zaxis=attr(title="Turn", range=[-0.1, 1.1])
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
        sol = solve(prob, Tsit5(), saveat=0.01)
        push!(solutions, sol)
    end
    return solutions
end

function find_limit_cycle(f, x, t, tol=5e-4)
    prob = ODEProblem(f, x, (0.0, t), params)
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

# Run the simulation and generate plots with more initial conditions
initial_conditions = make_hypersphere(0.1, 3, 10, [0.5, 0.5, 0.5])
tspan = (0.0, 1000.0)
solutions = run_simulation(name, initial_conditions, tspan, params)

# Generate roots from a grid search
grid_points = range(-1.0, stop=1.0, length=10)
roots = grid_search_roots(grid_points, params)
#println("Unique roots found: $roots")

# Find limit cycles
limit_cycles = []
limit_cycle_ics = [[-0.5290695,0.6016491, 0.8298353]]
for u0 in limit_cycle_ics
    cycle, _ = find_limit_cycle(vector_field!, u0, 30.0)
    if !isempty(cycle)
        push!(limit_cycles, cycle)
    end
end
#println("Limit cycles found: $limit_cycles")

# Plot limit cycle separately
if !isempty(limit_cycles)
    plot_limit_cycle(limit_cycles[1], name)
end

# Plot phase portrait with trajectories and limit cycles
plot_phase_portrait(roots, solutions, [], name, params)
