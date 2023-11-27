using Plots, IntervalSets

f = x -> exp(x)cos(x^2)

n = 7
x = range(0,1,n+1)


###
# Right
###

p = plot(legend=false, title="(Right-sided) Rectangular Rule")
for k = 1:n
    plot!([x[k:k+1]; x[k+1:-1:k]; x[k]], [0,0,f(x[k+1]),f(x[k+1]),0], color=:darkred, fill=(0,0.5,:red))
end; p
plot!(range(0,1,1000), f; linewidth=2, color=:blue)
savefig("slides/figures/slides1_rightrect.pdf")

###
# Left
###

p = plot(legend=false, title="(Left-sided) Rectangular Rule")
for k = 1:n
    plot!([x[k:k+1]; x[k+1:-1:k]; x[k]], [0,0,f(x[k]),f(x[k]),0], color=:darkred, fill=(0,0.5,:red))
end; p
plot!(range(0,1,1000), f; linewidth=2, color=:blue)
savefig("slides/figures/slides1_leftrect.pdf")

###
# Trapezium
###

p = plot(legend=false, title="Trapezium Rule")
for k = 1:n
    plot!([x[k:k+1]; x[k+1:-1:k]; x[k]], [0,0,f(x[k+1]),f(x[k]),0], color=:darkred, fill=(0,0.5,:red))
end; p
plot!(range(0,1,1000), f; linewidth=2, color=:blue)
savefig("slides/figures/slides1_trap.pdf")

