using Plots

function lagrange(𝐱, k, x)
    ret = one(x)
    n = length(𝐱)
    for j = 1:n
        if j ≠ k
            ret *= (x-𝐱[j])/(𝐱[k]-𝐱[j])
        end
    end
    ret
end

n = 4
𝐱 = range(0, 1; length=n) # evenly spaced points (BAD for interpolation)
g = range(0, 1; length=1000)
plot(g, lagrange.(Ref(𝐱), 1, g); label="ℓ₁"); scatter!(𝐱, lagrange.(Ref(𝐱), 1, 𝐱); label=nothing, color=:blue)
savefig("slides/figures/slides_regress_l1.pdf")

k = 2; plot!(g, lagrange.(Ref(𝐱), k, g); label="ℓ₂", color=:red); scatter!(𝐱, lagrange.(Ref(𝐱), k, 𝐱); label=nothing, color=:red)
k = 3; plot!(g, lagrange.(Ref(𝐱), k, g); label="ℓ₃", color=:green); scatter!(𝐱, lagrange.(Ref(𝐱), k, 𝐱); label=nothing, color=:green)
k = 4; plot!(g, lagrange.(Ref(𝐱), k, g); label="ℓ₄", color=:orange); scatter!(𝐱, lagrange.(Ref(𝐱), k, 𝐱); label=nothing, color=:orange)

savefig("slides/figures/slides_regress_l.pdf")