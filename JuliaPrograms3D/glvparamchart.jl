using Revise, Plots
using BifurcationKit
const BK = BifurcationKit

# vector field
function GLVvf(z, p)
	(;α, β) = p
	f, p, r = z
    df = f * (1 - f * f - α * p^2 - β * r * r)
    dp = p * (1 - p * p - α * r^2 - β * f * f)
    dr = r * (1 - r * r - α * f^2 - β * p * p)
	return [df, dp, dr]
end

# parameter values
par_tm = (α = .7, β = 0.7)

# initial condition
z0 = [0.99, 0.01, 0.01]
z1 = [0.01, 0.99, 0.01]
z2 = [0.01, 0.01, 0.99]
z =  [1/(1+par_tm.α+par_tm.β), 1/(1+par_tm.α+par_tm.β), 1/(1+par_tm.α+par_tm.β)]

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

scene = plot(br_β, vars=(:α,:β), legend=:topleft, range=(0, 3))

br_αβ = continuation(br_α, 1, (@optic _.β), opts_br; normC = norminf)

br_βα = continuation(br_β, 1, (@optic _.α), opts_br; normC = norminf)

# Hopf parameter chart
scene2 = plot(br_βα, vars=(:α,:β), legend=:topleft, range=(0, 3), 
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