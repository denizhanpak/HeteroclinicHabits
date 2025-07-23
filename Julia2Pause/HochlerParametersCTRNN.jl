using DifferentialEquations
using Plots
using Random
using StatsBase
using NLsolve  # Use NonlinearSolve instead of NLsolve
using LinearAlgebra  # Import norm function
#using PlotlyJS  # Import PlotlyJS for interactive plots
include("./HelpersFunctions.jl")
include("./DwellTime.jl")

function jacobian!(u, p)
    J = zeros(3, 3)
    ρ = p[1]  # Extract parameter matrix from parameters
    J[1, 1] = 1 - 3 * ρ[1, 1] * u[1]^2 - ρ[1, 2] * u[2]^2 - ρ[1, 3] * u[3]^2
    J[1, 2] = -2 * ρ[1, 2] * u[1] * u[2]
    J[1, 3] = -2 * ρ[1, 3] * u[1] * u[3]

    J[2, 1] = -2 * ρ[2, 1] * u[2] * u[1]
    J[2, 2] = 1 - 3 * ρ[2, 2] * u[2]^2 - ρ[2, 3] * u[3]^2 - ρ[2, 1] * u[1]^2
    J[2, 3] = -2 * ρ[2, 3] * u[2] * u[3]
    
    J[3, 1] = -2 * ρ[3, 1] * u[3] * u[1]
    J[3, 2] = -2 * ρ[3, 2] * u[3] * u[2]
    J[3, 3] = 1 - 3 * ρ[3, 3] * u[3]^2 - ρ[3, 1] * u[1]^2 - ρ[3, 2] * u[2]^2
end

function parameter_matrix(α, β, v)
    ρ = [
        α[1]/β[1] (α[1] - α[2]/v[2])/β[2] (α[1] + α[3])/β[3];
        (α[2] + α[1])/β[1]  α[2]/β[2] (α[2]-α[3]/v[3])/β[3];
        (α[3] - α[1]/v[1])/β[1] (α[3] + α[2])/β[2] α[3]/β[3]
    ]
    return ρ
end

function sigmoid(x; d=0, e=1)
    return 1 ./ (1 .+ exp.(-(x .- d) ./ e))
end

function vector_field!(du, u, p, t)
    ρ = p[1]
    α = p[2]
    ρ =Transpose(ρ)
    du[1] = -u[1] 
    du[1] = u[1] * (α[1] - ρ[1,1] * u[1]^2 - ρ[2,1] * u[2]^2 - ρ[3,1] * u[3]^2)
    du[2] = u[2] * (α[2] - ρ[1,2] * u[1]^2 - ρ[2,2] * u[2]^2 - ρ[3,2] * u[3]^2)
    du[3] = u[3] * (α[3] - ρ[1,3] * u[1]^2 - ρ[2,3] * u[2]^2 - ρ[3,3] * u[3]^2)
end

function noise_term!(du, u, p, t)
    n = 1e-15
    du[1] = n
    du[2] = n
    du[3] = n
end


name = "excitable_network"
# #mu = 1
# #a = 1.0
# #b = 0.55
# #c = 1.5
α = [1., 0.5, 1.] .* 1.
β = [1.0, 1.0, 1.0] .* 1.
v = [1.5, 1.5, 1.5] .* 3.
params = (parameter_matrix(α, β, v), α)
ds = DS(3, vector_field!, jacobian!, noise_term!, x->x^2, params)
t = 100.0
# Run the simulation and generate plots with more initial conditions
initial_conditions = make_hypersphere(0.01, 3, 1, [0.53, 0.53, 0.53])
initial_conditions = [[0.1,0.1,0.9]]
tspan = (0.0, t)
#plot_time_series(ds, initial_conditions, tspan, name)
pl, sol = plot_time_series(ds, initial_conditions, tspan, name)
pl
#sol = solve_SDE(initial_conditions[1], ds, tspan)
#sol = solve_ODE(initial_conditions[1], ds, tspan)
#plot(sol)
adt = average_dwell_times(sol[1], [[1,0,0],[0,1,0],[0,0,1]])
rv=[adt[i] / minimum(values(adt)) for i in [1, 2, 3]]