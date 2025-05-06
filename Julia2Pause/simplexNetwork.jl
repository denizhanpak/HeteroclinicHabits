using DifferentialEquations
using Plots
using Random
using StatsBase
using NLsolve  # Use NonlinearSolve instead of NLsolve
using LinearAlgebra  # Import norm function
#using PlotlyJS  # Import PlotlyJS for interactive plots
include("./HelpersFunctions.jl")


function jacobian!(u, p)
    J = zeros(3, 3)
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

function noise_term!(du, u, p, t)
    mu, a, b, c = p
    n = 0.1
    du[1] = n
    du[2] = n
    du[3] = n
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

name = "excitable_network"
# #mu = 1
# #a = 1.0
# #b = 0.55
# #c = 1.5
params = (1.0, 1.0, 1.01, 2)
ds = DS(3, vector_field!, jacobian!, noise_term!, x->x^2, params)
t = 200.0
# Run the simulation and generate plots with more initial conditions
initial_conditions = make_hypersphere(0.1, 3, 1, [0.5, 0.5, 0.5])
tspan = (0.0, t)
plot_time_series(ds, initial_conditions, tspan, name)

name = "heteroclinic_network"
params = (1.0, 1.0, 0.6, 2)
ds = DS(3, vector_field!, jacobian!, noise_term!, x->x^2, params)

# Run the simulation and generate plots with more initial conditions
initial_conditions = make_hypersphere(0.1, 3, 1, [0.5, 0.5, 0.5])
tspan = (0.0, t)
plot_time_series(ds, initial_conditions, tspan, name)
exit()

# Generate roots from a grid search
roots = length(grid_search_roots(ds))
println("Unique roots found: $roots")
plot_phase_portrait(ds, initial_conditions)

# # Find limit cycles
# limit_cycles = []
# limit_cycle_ics = [[-0.5290695,0.6016491, 0.8298353]]
# for u0 in limit_cycle_ics
#     cycle, _ = find_limit_cycle(vector_field!, u0, 30.0)
#     if !isempty(cycle)
#         push!(limit_cycles, cycle)
#     end
# end
# #println("Limit cycles found: $limit_cycles")

# # Plot limit cycle separately
# if !isempty(limit_cycles)
#     plot_limit_cycle(limit_cycles[1], name)
# end

# # Plot phase portrait with trajectories and limit cycles

