using Plots

# ...existing code...

# Set up the plot with the specified axis ranges and aspect ratio
plot(xlims=(0, 3), ylims=(0, 3), legend=false, aspect_ratio=:equal)

# Add a black line parallel to the x-axis from (1, 3)
plot!([1, 3], [1, 1], color=:black)

# Add a black line parallel to the y-axis from (1, 3)
plot!([1, 1], [1, 3], color=:black)
plot!([0, 2], [2, 0], color=:black)

# Save the plot as glv_plot.png
savefig("glv_plot.png")

# Display the plot
display(plot)
