using FFTW, Plots
f = θ -> 2/(2-exp(im*θ))

for n =  [5, 10]
    θ = range(0, 2π, n+1)[1:end-1]

    f̂ = fft(f.(θ))/n

    @test n*ifft(f̂) ≈ f.(θ)


    m = 1000
    g = range(0, 2π, m+1)
    plot(g, real.(f.(g)); label="Re(f)", linewidth=2, title="Real part, n = $n")
    v = m*ifft([f̂; zeros(m-length(f̂))]); v = [v; v[1]]; plot!(g, real.(v); label="Re(fₙ)", linewidth=2)
    scatter!(θ, real.(f.(θ)); label=nothing)
    savefig("slides/figures/realfft_analytic_$n.pdf")

    plot(g, imag.(f.(g)); label="Im(f)", linewidth=2, title="Imag part, n = $n")
    v = m*ifft([f̂; zeros(m-length(f̂))]); v = [v; v[1]]; plot!(g, imag.(v); label="Im(fₙ)", linewidth=2)
    scatter!(θ, imag.(f.(θ)); label=nothing)
    savefig("slides/figures/imagfft_analytic_$n.pdf")
end

f = θ -> 2/(25cos(θ-1)^2 + 1)

for n =  [5, 10, 100]
    θ = range(0, 2π, n+1)[1:end-1]

    f̂ = fft(f.(θ))/n

    @test n*ifft(f̂) ≈ f.(θ)


    m = 1000
    g = range(0, 2π, m+1)
    plot(g, real.(f.(g)); label="Re(f)", linewidth=2, title="Real part, n = $n")
    v = m*ifft([f̂; zeros(m-length(f̂))]); v = [v; v[1]]; plot!(g, real.(v); label="Re(fₙ)", linewidth=2)
    scatter!(θ, real.(f.(θ)); label=nothing)
    savefig("slides/figures/realfft_runge_$n.pdf")

    plot(g, imag.(f.(g)); label="Im(f)", linewidth=2, title="Imag part, n = $n")
    v = m*ifft([f̂; zeros(m-length(f̂))]); v = [v; v[1]]; plot!(g, imag.(v); label="Im(fₙ)", linewidth=2)
    scatter!(θ, imag.(f.(θ)); label=nothing)
    savefig("slides/figures/imagfft_runge_$n.pdf")
end

using LaTeXStrings

for n =  [5, 11, 101]
    θ = range(0, 2π, n+1)[1:end-1]

    f̂ = fft(f.(θ))/n

    @test n*ifft(f̂) ≈ f.(θ)


    M = 500
    N = 2M+1
    m = n ÷ 2
    
    
    g = range(0, 2π, N+1)
    plot(g, real.(f.(g)); label=L"Re(f)", linewidth=2, title="Real part, n = $n")
    v = N*ifft([f̂[1:m+1]; zeros(N - n); f̂[m+2:end]]); v = [v; v[1]]; plot!(g, real.(v); label=L"Re(f_{-m:m})", linewidth=2)
    scatter!(θ, real.(f.(θ)); label=nothing)
    savefig("slides/figures/realfft_m_runge_$n.pdf")

    plot(g, imag.(f.(g)); label="Im(f)", linewidth=2, title="Imag part, n = $n", ylims=(-5E-16,5E-16))
    v = N*ifft([f̂[1:m+1]; zeros(N - n); f̂[m+2:end]]); v = [v; v[1]]; plot!(g, imag.(v); label="Im(fₙ)", linewidth=2)
    scatter!(θ, imag.(f.(θ)); label=nothing)
    savefig("slides/figures/imagfft_m_runge_$n.pdf")
end