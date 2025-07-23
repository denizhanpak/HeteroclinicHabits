using Revise, Plots
using BifurcationKit
const BK = BifurcationKit

# vector field
function GLVvf(z, p)
	(;α, β) = p
	f, p1, r, p2 = z
    df = f * (1 - f - α * p2 - β * p1)
    dp1 = p1 * (1 - p1 - α * f - β * r)
    dr = r * (1 - r - α * p1 - β * p2)
    dp2 = p2 * (1 - p2 - α * r - β * f)
	return [df, dp1, dr, dp2]
end

name = "heteroclinic_network"
params = (1.0, 1.0, 0.6, 2)
ds = DS(4, vector_field!, jacobian!, noise_term!, x->x^2, params)

# Run the simulation and generate plots with more initial conditions
initial_conditions = make_hypersphere(0.1, 4, 1, [0.5, 0.5, 0.5, 0.5])
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


# parameter values
par_tm = (α = .1, β = 0.1)

# initial condition
z0 = [0.99, 0.01, 0.01, 0.01]
z1 = [0.01, 0.99, 0.01, 0.01]
z2 = [0.01, 0.01, 0.99, 0.01]
z =  [1/(1+par_tm.α+par_tm.β), 1/(1+par_tm.α+par_tm.β), 1/(1+par_tm.α+par_tm.β), 1/(1+par_tm.α+par_tm.β)]

# record the solution
rec_glv(x, p; k...) = (f = x[1], p = x[2], r = x[3])#, α = p[1], β = p[2])


# Bifurcation Problem
prob_α = BifurcationProblem(GLVvf, z, par_tm, (@optic _.α);
	record_from_solution = rec_glv,)

prob_β = BifurcationProblem(GLVvf, z, par_tm, (@optic _.β);
	record_from_solution = rec_glv,)

# continuation options, we limit the parameter range for E0
opts_br = ContinuationPar(p_min = 0.0, p_max = 3.0, ds = 0.001, dsmax = 0.008)

# continuation of equilibria
br_α = continuation(prob_α, PALC(), opts_br; normC = norminf)

br_β = continuation(prob_β, PALC(), opts_br; normC = norminf)

scene = plot(br_α, legend=:topleft)
scene = plot(br_β, legend=:topleft)

br_αβ = continuation(br_α, 1, (@optic _.β), opts_br; normC = norminf)
br_βα = continuation(br_β, 1, (@optic _.α), opts_br; normC = norminf)

# Hopf parameter chart
scene2 = plot(br_βα, vars=(:α,:β), legend=:topleft, range=(0, 3), 
	title="Parameter chart of α and β", xlabel="α", ylabel="β", color="blue")
plot!(br_αβ, vars=(:α,:β), legend=:topleft, color="blue", aspect_ratio=1, 
	ylims=(0, 3), xlims=(0, 3), label="Hopf Bifurcation")

# Saddle bifurcation
par_tm = (α = .1, β = .1)
prob_α = BifurcationProblem(GLVvf, z0, par_tm, (@optic _.α);
	record_from_solution = rec_glv,)

prob_β = BifurcationProblem(GLVvf, z0, par_tm, (@optic _.β);
	record_from_solution = rec_glv,)

# continuation options, we limit the parameter range for E0
opts_br = ContinuationPar(p_min = 0.0, p_max = 3.0, ds = 0.001, dsmax = 0.008)

# continuation of equilibria
sn_α = continuation(prob_α, PALC(), opts_br; normC = norminf)
sn_β = continuation(prob_β, PALC(), opts_br; normC = norminf)
sn_αβ = continuation(sn_α, 1, (@optic _.β), opts_br; normC = norminf)
sn_βα = continuation(sn_β, 1, (@optic _.α), opts_br; normC = norminf)

plot!(sn_αβ, vars=(:α,:β), legend=:topleft, color="red")
plot!(sn_βα, vars=(:α,:β), legend=:topleft, color="red")