using DifferentialEquations
using Plots
using Random
using StatsBase
using NLsolve  # Use NonlinearSolve instead of NLsolve
using LinearAlgebra  # Import norm function
using Statistics
#using PlotlyJS  # Import PlotlyJS for interactive plots
include("./HelpersFunctions.jl")
include("./DwellTime.jl")

function jacobian!(u, p)
    J = zeros(4, 4)  # Update to 4x4 matrix
    ρ = p[1]  # Extract parameter matrix from parameters
    J[1, 1] = 1 - 3 * ρ[1, 1] * u[1]^2 - ρ[1, 2] * u[2]^2 - ρ[1, 3] * u[3]^2 - ρ[1, 4] * u[4]^2
    J[1, 2] = -2 * ρ[1, 2] * u[1] * u[2]
    J[1, 3] = -2 * ρ[1, 3] * u[1] * u[3]
    J[1, 4] = -2 * ρ[1, 4] * u[1] * u[4]

    J[2, 1] = -2 * ρ[2, 1] * u[2] * u[1]
    J[2, 2] = 1 - 3 * ρ[2, 2] * u[2]^2 - ρ[2, 3] * u[3]^2 - ρ[2, 4] * u[4]^2 - ρ[2, 1] * u[1]^2
    J[2, 3] = -2 * ρ[2, 3] * u[2] * u[3]
    J[2, 4] = -2 * ρ[2, 4] * u[2] * u[4]

    J[3, 1] = -2 * ρ[3, 1] * u[3] * u[1]
    J[3, 2] = -2 * ρ[3, 2] * u[3] * u[2]
    J[3, 3] = 1 - 3 * ρ[3, 3] * u[3]^2 - ρ[3, 4] * u[4]^2 - ρ[3, 1] * u[1]^2 - ρ[3, 2] * u[2]^2
    J[3, 4] = -2 * ρ[3, 4] * u[3] * u[4]

    J[4, 1] = -2 * ρ[4, 1] * u[4] * u[1]
    J[4, 2] = -2 * ρ[4, 2] * u[4] * u[2]
    J[4, 3] = -2 * ρ[4, 3] * u[4] * u[3]
    J[4, 4] = 1 - 3 * ρ[4, 4] * u[4]^2 - ρ[4, 1] * u[1]^2 - ρ[4, 2] * u[2]^2 - ρ[4, 3] * u[3]^2
end

function parameter_matrix(α, β, v)
    ρ = [
        α[1]/β[1] (α[1] - α[2]/v[2])/β[2] (α[1] + α[3])/β[3] (α[1] + α[4])/β[4];
        (α[2] + α[1])/β[1]  α[2]/β[2] (α[2] - α[3]/v[3])/β[3] (α[2] + α[4])/β[4];
        (α[3] + α[1])/β[1] (α[3] + α[2])/β[2] α[3]/β[3] (α[3] - α[4]/v[4])/β[4];
        (α[4] - α[1]/v[1])/β[1] (α[4] + α[2])/β[2] (α[4] + α[3])/β[3] α[4]/β[4]
    ]
    return ρ
end

function vector_field!(du, u, p, t)
    ρ = p[1]
    α = p[2]
    ρ = Transpose(ρ)
    du[1] = u[1] * (α[1] - ρ[1,1] * u[1]^2 - ρ[2,1] * u[2]^2 - ρ[3,1] * u[3]^2 - ρ[4,1] * u[4]^2)
    du[2] = u[2] * (α[2] - ρ[1,2] * u[1]^2 - ρ[2,2] * u[2]^2 - ρ[3,2] * u[3]^2 - ρ[4,2] * u[4]^2)
    du[3] = u[3] * (α[3] - ρ[1,3] * u[1]^2 - ρ[2,3] * u[2]^2 - ρ[3,3] * u[3]^2 - ρ[4,3] * u[4]^2)
    du[4] = u[4] * (α[4] - ρ[1,4] * u[1]^2 - ρ[2,4] * u[2]^2 - ρ[3,4] * u[3]^2 - ρ[4,4] * u[4]^2)
end

function noise_term!(du, u, p, t)
    n = 1e-4
    du[1] = n
    du[2] = n
    du[3] = n
    du[4] = n
end

function check_condition(a,v)
    for i in eachindex(a)
        j = (i + 1) % length(a) + 1
        grate = a[j]
        ratio = a[i] / v[i]
        if grate < ratio
            println("$grate < $ratio")
            return false
        end
    end
    return true
end

name = "excitable_network"
α = [1/25, 1.3, 1/18, 1.3] .* 1.  # Add fourth state parameter
α = [1/22, 1.6, 1/10, 1.4] .* 1.  # state for mock ctrnn data
β = [1.0, 1.0, 1.0, 1.0] .* 1.  # Add fourth state parameter
v = [3, 6, 1.5, 1.5] .* 2.  # Add fourth state parameter
check_condition(α,v)  # Check condition for the new parameters


params = (parameter_matrix(α, β, v), α)
ds = DS(4, vector_field!, jacobian!, noise_term!, x->x^2, params)  # Update to 4 dimensions
t = 10000.0
# Run the simulation and generate plots with more initial conditions
initial_conditions = make_hypersphere(0.01, 4, 1, [0.53, 0.53, 0.53, 0.53])  # Update to 4 dimensions
initial_conditions = [[0.1, 0.1, 0.9, 0.1]]  # Update to 4 dimensions
tspan = (0.0, t)
#plot_time_series(ds, initial_conditions, tspan, name)
pl, sol = plot_time_series(ds, initial_conditions, tspan, name)
pl
#sol = solve_SDE(initial_conditions[1], ds, tspan)
#sol = solve_ODE(initial_conditions[1], ds, tspan)
#plot(sol)
adt = average_dwell_times(sol[1], [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
rv=[adt[i] / minimum(values(adt)) for i in [1, 2, 3, 4]]

dts = evaluate_dwell_times(ds, (0, 20000), 5)
mean(dts, dims=2)
std(dts, dims=2)