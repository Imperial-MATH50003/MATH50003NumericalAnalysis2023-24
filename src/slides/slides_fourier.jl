using FFTW, Plots, Test


f = θ -> exp(θ/(2π))
using LaTeXStrings

for n =  [5, 11, 101]
    θ = range(0, 2π, n+1)[1:end-1]

    f̂ = fft(f.(θ))/n

    @test n*ifft(f̂) ≈ f.(θ)


    M = 500
    N = 2M+1
    m = n ÷ 2
    
    
    g = range(0, 2π, N+1)
    plot(g, real.(f.(g)); label=L"f", linewidth=2, title="Real part, n = $n")
    v = N*ifft([f̂[1:m+1]; zeros(N - n); f̂[m+2:end]]); v = [v; v[1]]; plot!(g, real.(v); label=L"Re(f_{-m:m})", linewidth=2)
    scatter!(θ, real.(f.(θ)); label=nothing)
    savefig("slides/figures/realfft_m_exp_$n.pdf")
end


using ClassicalOrthogonalPolynomials
for n =  [5, 11, 101]
    T_n =chebyshevt(0..1)[:,1:n]
    f_n = expand(T_n, exp)
    g = range(0, 1, N+1)
    plot(g, exp.(g), linewidth=2, title="n = $n", label=L"f")
    plot!(g, f_n[g], linewidth=2, label=L"f_n")
        
    x = ClassicalOrthogonalPolynomials.grid(T_n)
    scatter!(x, exp.(x); label=nothing)
    savefig("slides/figures/cheb_exp_$n.pdf")
end