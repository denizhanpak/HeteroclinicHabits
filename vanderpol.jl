using DifferentialEquations
using Plots
using FFTW

# Define the Van der Pol oscillator differential equation
function vanderpol!(du, u, p, t)
    μ = p
    du[1] = u[2]
    du[2] = μ * (1 - u[1]^2) * u[2] - u[1]
end

# List of initial conditions and parameters
initial_conditions = [[2.0, 0.0], [1.0, 1.0], [0.0, 2.0]]
#initial_conditions = []
tspan = (0.0, 20.0)
p = 2.0

# Solve the problem for each initial condition
solutions = []
lengths = []
for u0 in initial_conditions
    prob = ODEProblem(vanderpol!, u0, tspan, p)
    sol = solve(prob, abstol = 1e-11, dt = 0.001, reltol = 1e-11)
    push!(solutions, sol)
    push!(lengths, length(sol.t))
end 

sol = solutions[1,:]



# Create a phase space GIF with each trajectory in a different color
if length(solutions) != 0
    anim = @animate for i in 1:min(lengths...)
        plot(xlim=(-4, 4), ylim=(-4, 4), xlabel="x", ylabel="y", title="Phase Space of Van der Pol Oscillator")
        for (idx, sol) in enumerate(solutions)
            plot!(sol[1, 1:i], sol[2, 1:i], label="Trajectory $(idx)")
        end
    end
    gif(anim, "vanderpol_phase_space.gif", fps=15)
end
exit()

# Generate a vector field plot
range = 5
x = -range:0.5:range
y = -range:0.5:range
u = zeros(length(x), length(y))
v = zeros(length(x), length(y))
for i in eachindex(x)
    for j in eachindex(y)
        du = zeros(2)
        vanderpol!(du, [x[i], y[j]], p, 0.0)
        u[i, j] = du[1]
        v[i, j] = du[2]
    end
end

quiver(x, y, quiver=(u, v), xlabel="x", ylabel="y", title="Vector Field of Van der Pol Oscillator")
savefig("vanderpol_vector_field.png")

