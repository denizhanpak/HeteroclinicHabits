using DifferentialEquations, Plots

# Parameters
rho = 2
K = 1
omega_1 = 1
omega_2 = 1
parameters = (rho, K, omega_1, omega_2)

# Initial conditions: [phase velocity 1, phase 1, phase velocity 2, phase 2]
u0 = [0.0, 0.0]

function body(du, u, p, t)
    rho, K, omega_1, omega_2 = p
    # Phase velocity 1, phase 1, phase velocity 2, phase 2
    du[1] = omega_1 + K/2 * sin(u[2] - u[1])
    du[2] = omega_2 + K/2 * sin(u[1] - u[2])
end

# Time span
tspan = (0.0, 100.0)

# Solve the system of ODEs
prob = ODEProblem(body, u0, tspan, parameters)
sol = solve(prob, Tsit5(), reltol=1e-8, abstol=1e-8)

# Visualization function
function visualize_solution(sol)
    plot_title = "Phase Trajectories of Two Coupled Oscillators"
    plot(sol.t, sin.(sol[1, :]), label="Phase 1", xlabel="Time", ylabel="Phase", title=plot_title)
    plot!(sol.t, sin.(sol[2, :]), label="Phase 2")
    savefig("two_oscillators_trajectories.png")
end

# Run visualization
visualize_solution(sol)
