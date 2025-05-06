using Revise
using Plots
using BifurcationKit
using DifferentialEquations
const BK = BifurcationKit

function sigmoid(x)
	return 1 / (1 + exp(-x))
end

function CTRNNvf(z, p)
	(;we, wi) = p
	f, p, r = z
	we *= 11
	wi *= 11
	ws = 2
	df = -f + ws * sigmoid(f) - we * sigmoid(p) - wi * sigmoid(r)
	dp = -p + ws * sigmoid(p) - we * sigmoid(r) - wi * sigmoid(f)
	dr = -r + ws * sigmoid(r) - we * sigmoid(f) - wi * sigmoid(p)
	return [df, dp, dr]
end

function CTRNNvf2(du, u, p, t)
	(;we, wi) = p
	f, p, r = u
	we *= 11
	wi *= 11
	ws = 2
	du[1] = -f + ws * sigmoid(f) - we * sigmoid(p) - wi * sigmoid(r)
	du[2] = -p + ws * sigmoid(p) - we * sigmoid(r) - wi * sigmoid(f)
	du[3] = -r + ws * sigmoid(r) - we * sigmoid(f) - wi * sigmoid(p)
end

# parameter values
par_tm = (we = .8, wi = .2)

# initial condition
z0 = [0.99, 0.01, 0.01]
z1 = [0.01, 0.99, 0.01]
z2 = [0.01, 0.01, 0.99]
z =  [1.5, 1.4, 1.5]

# Solve and plot for initial condition (0, 0, 0)
z_init = [1.0, 2.0, 3.0]
tspan = (0.0, 100.0)  # Define the time span
prob2 = ODEProblem(CTRNNvf2, z0, tspan, par_tm)
sol2 = solve(prob2, Tsit5())

plot(sol2, title="CTRNN Solution", xlabel="Time", ylabel="State Variables")

# record the solution
rec_ctrnn(x, p; k...) = (f = x[1], p = x[2], r = x[3])#, α = p[1], β = p[2])


# Bifurcation Problem
prob_we = BifurcationProblem(CTRNNvf, z, par_tm, (@optic _.we);
	record_from_solution = rec_ctrnn,)

prob_wi = BifurcationProblem(CTRNNvf, z, par_tm, (@optic _.wi);
	record_from_solution = rec_ctrnn,)

# continuation options, we limit the parameter range for E0
opts_br = ContinuationPar(p_min = 0.0, p_max = 1.1, ds = 0.0001, dsmax = 0.001)

diagram = bifurcationdiagram(prob_we, PALC(), 4, opts_br,)

plot(diagram)

# continuation of equilibria
br_we = continuation(prob_we, PALC(), opts_br; normC = norminf)

br_wi = continuation(prob_wi, PALC(), opts_br; normC = norminf)

scene = plot(br_we, legend=:topleft)
scene = plot(br_wi, legend=:topleft)

scene = plot(br_we, legend=:topleft, range=(0, 1))

br_ei = continuation(br_we, 1, (@optic _.wi), opts_br; normC = norminf)

br_ie = continuation(br_wi, 1, (@optic _.we), opts_br; normC = norminf)

# Hopf parameter chart
scene2 = plot(br_, vars=(:α,:β), legend=:topleft, range=(0, 3), 
	title="Parameter chart of α and β", xlabel="α", ylabel="β", color="blue")
plot!(br_αβ, vars=(:α,:β), legend=:topleft, color="blue")

# Saddle bifurcation
prob_α = BifurcationProblem(GLVvf, z0, par_tm, (@optic _.α);
	record_from_solution = rec_glv,)

prob_β = BifurcationProblem(GLVvf, z0, par_tm, (@optic _.β);
	record_from_solution = rec_glv,)

# continuation options, we limit the parameter range for E0
opts_br = ContinuationPar(p_min = 0.0, p_max = 3.0, ds = 0.001, dsmax = 0.008)

# continuation of equilibria
sn_α = continuation(prob_α, PALC(), opts_br; normC = norminf)
sn_β = continuation(prob_β, PALC(), opts_br; normC = norminf)
sn_αβ = continuation(br_α, 1, (@optic _.β), opts_br; normC = norminf)
sn_βα = continuation(br_β, 1, (@optic _.α), opts_br; normC = norminf)

plot!(sn_αβ, vars=(:α,:β), legend=:topleft, color="red")
plot!(sn_βα, vars=(:α,:β), legend=:topleft, color="red")