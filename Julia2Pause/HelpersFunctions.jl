struct DS
    dimensions::Int
    vector_field::Function
    jacobian::Function
    noise::Function
    plot_transform::Function
    parameters::Tuple
end

function find_roots(u0::Vector, ds::DS)
    if length(u0) != ds.dimensions
        throw(ArgumentError("Initial condition must have the same length as the number of dimensions"))
    end

    function f!(F, x)
        ds.vector_field(F, x, ds.parameters, 0.0)
    end
    function j!(J, x)
        ds.jacobian(J, x, ds.parameters)
    end
    result = nlsolve(f!, j!, u0)
    
    return result.zero
end


function classify_stability(solution::Vector, ds::DS)
    J = ds.jacobian(solution, ds.parameters)
  
    eigenvalues = eigvals(J)
    println(eigenvalues)
    if any(abs.(real.(eigenvalues)) .< 1e-6)
        return :center, eigenvalues
    elseif all(real.(eigenvalues) .< 0)
        return :stable, eigenvalues
    elseif all(real.(eigenvalues) .> 0)
        return :unstable, eigenvalues
    else
        return :saddle, eigenvalues
    end
end

function remove_duplicate_roots(roots::Vector, threshold::Real=1e-6)
    unique_roots = []
    for root in roots
        if all(norm(root .- r) >= threshold for r in unique_roots)
            push!(unique_roots, root)
        end
    end
    return unique_roots
end

function grid_search_roots(ds::DS, threshold::Real=1e-6, mesh_range::Tuple=(0,1.1), mesh_size::Int=10)
    grid_points = collect(range(mesh_range[1], mesh_range[2], length=mesh_size))
    roots = []
    for x in grid_points
        for y in grid_points
            for z in grid_points
                u0 = [x, y, z]
                root = find_roots(u0, ds)
                if all(isfinite, root)
                    push!(roots, root)
                end
            end
        end
    end
    unique_roots = remove_duplicate_roots(roots, threshold)
    return unique_roots
end

function run_simulation(ds::DS, initial_conditions::Vector, tspan::Tuple=(0.0, 50.0))
    solutions = []
    if ds.noise === nothing
        for u0 in initial_conditions
            prob = ODEProblem(ds.vector_field, u0, tspan, ds.parameters)
            sol = solve(prob, Tsit5(), saveat=0.01)
            push!(solutions, sol)
        end
    else
        for u0 in initial_conditions
            prob = SDEProblem(ds.vector_field, ds.noise, u0, tspan, ds.parameters)
            sol = solve(prob, saveat=0.01)
            push!(solutions, sol)
        end
    end
    return solutions
end

function plot_time_series(ds::DS, solution_ICs::Vector, tspan::Tuple=(0.0,50.0), name::String="time_series")
    solutions = run_simulation(ds, solution_ICs, tspan)

    for (i, sol) in enumerate(solutions)
        time = sol.t
        for j in 1:ds.dimensions
            Plots.plot!(time, ds.plot_transform.(sol[j, :]), label="$i dim_$j", linewidth=0.1)
        end
    end
    Plots.savefig("$(name)_time_series.png")
end 

function plot_phase_portrait(ds::DS, solution_ICs::Vector, root_range::Tuple=(0.1,1.1), mesh_size::Int=10, limit_cycle_ics::Vector=[], name::String="phase_portrait")
    #plot the roots
    roots = grid_search_roots(ds, 1e-6, root_range, mesh_size)
    colors = []
    hovertexts = []
    trajectories = []
    for root in roots
        stability, eigenvalues = classify_stability(root, ds)
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
    
    #plot the trajectories
    solutions = run_simulation(ds, solution_ICs)
    for (i, sol) in enumerate(solutions)
        push!(trajectories, PlotlyJS.scatter3d(x=sol[1, :], y=sol[2, :], z=sol[3, :], mode="lines", line=attr(width=0.4), name="trajectory $i"))
    end

    #plot the limit cycles
    limit_cycles = run_simulation(ds, limit_cycle_ics)
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


function make_hypersphere(r=0.5, d=3, n=10,o=nothing)
    if o === nothing
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

function solve_ODE(ic::Vector, ds::DS, tspan::Tuple=(0.0, 50.0))
    prob = ODEProblem(ds.vector_field, ic, tspan, ds.parameters)
    sol = solve(prob, Tsit5(), saveat=0.01)
    return sol
end

function plot_saddles(point::Vector, ds::DS)
    J = zeros(ds.dimensions, ds.dimensions)
    jac = ds.jacobian(J, point, ds.parameters)
    eigenvalues, eigenvectors = eigen(jac)
    offset = 0.005
    trajectories = []
    for i in eachindex(eigenvalues)
        eigenvalue = real.(eigenvalues[i])
        eigenvector = real.(eigenvectors[:, i])
        time = 50
        u0 = root .+ offset * abs.(eigenvector)
        # Add a point at u0 for debugging
        #push!(trajectories, PlotlyJS.scatter3d(x=[u0[1]], y=[u0[2]], z=[u0[3]], mode="markers", marker=attr(size=5, color="black"), name="u0"))
        if real(eigenvalue) > 0
            sol = solve_ODE(u0, ds, (0.0, time))
            push!(trajectories, PlotlyJS.scatter3d(x=sol[1, :], y=sol[2, :], z=sol[3, :], mode="lines", line=attr(color="red", width=0.6), name="unstable"))
        elseif real(eigenvalue) < 0
            sol = solve_ODE(u0, ds, (0.0, time))
            push!(trajectories, PlotlyJS.scatter3d(x=sol[1, :], y=sol[2, :], z=sol[3, :], mode="lines", line=attr(color="red", width=0.6), name="unstable"))
        end
    end
    return trajectories
end


