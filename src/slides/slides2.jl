using Plots

f = x -> exp(x)cos(x^2)

n = 7
x = range(0,1,n+1)


###
# Right
###

x = 0.5
for h = (0.3, 0.1, 0.01, 0.0000000000000001)
    p = plot(title="(Right-sided) Divided Difference")
    plot!(range(0,1,1000), f; linewidth=2, color=:blue, label="f")
    scatter!([x,x+h], f.([x,x+h]); color=:darkred, label=nothing)
    # (f(x+h)-f(x))/h * (t-x) + f(x)
    plot!([0,1], [(f(x+h)-f(x))/h * (-x) + f(x), (f(x+h)-f(x))/h * (1-x) + f(x)], color=:darkred, linewidth=2, label="h = $h")
    p
    savefig("slides/figures/slides2_rightdiff_h=$h.pdf")
end


