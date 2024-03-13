using Plots, LaTeXStrings
p = plot()
for k = 1:5
    plot!(ChebyshevT()[:,k]; label = latexstring("T_{$(k-1)}"), linewidth=2)
end; p
savefig("slides/figures/chebyshevt.pdf")


p = plot()
for k = 1:5
    plot!(ChebyshevU()[:,k]; label = latexstring("U_{$(k-1)}"), linewidth=2)
end; p
savefig("slides/figures/chebyshevu.pdf")


p = plot()
for k = 1:5
    plot!(Legendre()[:,k]; label = latexstring("P_{$(k-1)}"), linewidth=2)
end; p
savefig("slides/figures/legendrep.pdf")