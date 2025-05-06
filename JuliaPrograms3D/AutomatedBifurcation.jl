using Revise, Plots
using BifurcationKit

Fbp(u, p) = @. -u * (p + u * (2-5u)) * (p -.15 - u * (2+20u))

# bifurcation problem
prob = BifurcationProblem(Fbp, [0.0], -0.2,
	# specify the continuation parameter
	(@optic _);
	record_from_solution = (x, p; k...) -> x[1])

# options for newton
# we reduce a bit the tolerances to ease automatic branching
opt_newton = NewtonPar(tol = 1e-9)

# options for continuation
opts_br = ContinuationPar(dsmin = 0.001, dsmax = 0.005, ds = 0.001,
	newton_options = opt_newton,
	nev = 1,
	# parameter interval
	p_min = -1.0, p_max = .3,
	# detect bifurcations with bisection method
	# we increase here the precision for the detection of
	# bifurcation points
	n_inversion = 8)

diagram = bifurcationdiagram(prob, PALC(),
	# very important parameter. This specifies the maximum amount of recursion
	# when computing the bifurcation diagram. It means we allow computing branches of branches
	# at most in the present case.
	2,
	opts_br,
)

# You can plot the diagram like
plot(diagram; putspecialptlegend=false, markersize=2, plotfold=false, title = "#branches = $(size(diagram))")