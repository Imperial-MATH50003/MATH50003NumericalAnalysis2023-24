using Plots, FastGaussQuadrature, LinearAlgebra

f = x -> 1/(25(x-1/2)^2+1)
function trapeziumrule(f, n)
    ret = f(-1)/2
    for j = 1:n-1 # j skips first and lest point
        ret = ret + f(-1+2j/n)
    end
    ret = ret + f(1)/2
    2ret/n
end
rectrule(n) = range(-1,1; length=n+1)[1:end-1], fill(2/n,n)
traprule(m) = (n = m-1; (range(-1,1; length=n+1), [1/(2n); fill(1/n,n-1); 1/(2n)]*2))

ex = (1/5)*(atan(5/2) + atan(15/2)) 
ns = 2:100
plot(ns, [((x,w) = rectrule(n); abs(dot(w,f.(x)) - ex)) for n = ns]; yscale=:log10, linewidth=2, label="Left-Rectangular", legend=:bottomleft, xlabel="number of points", ylabel="Error", yticks=10.0 .^ (-16:0))
plot!(ns, [((x,w) = traprule(n); abs(dot(w,f.(x)) - ex)) for n = ns]; yscale=:log10, linewidth=2, label="Trapezium")
nanabs(x) = x == 0 ? NaN : abs(x)
plot!([((x,w) = gausslegendre(n); nanabs(dot(w,f.(x)) - ex)) for n = ns]; yscale=:log10, linewidth=2, label="Gauss")

savefig("slides/figures/gausserror.pdf")

