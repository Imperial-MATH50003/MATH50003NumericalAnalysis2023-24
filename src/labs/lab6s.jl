# # MATH50003 (2022–23)
# # Lab 6: III.3 Cholesky Factorisation and III.4 Polynomial Regression

# In this lab we explore using LU, PLU and Cholesky factorisations, and
# implement algorithms for computing a Cholesky factorisation.

# **Learning Outcomes**
#
# Mathematical knowledge:
#
# 1. Cholesky and reverse Cholesky factorisations, including for banded matrices.
# 2. Vandermonde matrices and least squares.
#
# Coding knowledge:
#
# 1. Using the `lu` and `cholesky` functions.
# 2. Solving least squares problems via `\`

# We load the following packages

using LinearAlgebra, Plots, BenchmarkTools, Test

# ### III.3 LU and Cholesky Factorisations

# LU, PLU and Cholesky factorisations are closely related
# matrix factorisations that reduce a square matrix to a product of
# lower and upper triangular matrices and possibly with a permutation.
# We will only focus on the practical usage of LU and PLU, whilst digging into the
# details of implementing and generalising a Cholesky factorisation.

# 

# ### III.3.1 LU Factorisation


# If $A ∈ 𝔽^{n × n}$ is a square matrix where $𝔽$ is a field ($ℝ$ or $ℂ$)
# then we can find lower and upper triangular matrices $L,U ∈ 𝔽^{n × n}$ such that 
# $$
# A = LU.
# $$
# This is equivalent to Gaussian elimination but we will only focus on practical usage.
# This factorisation can be computed using the `lu` function, but as the default is a PLU factorisation we add a flag
# telling it not to use pivoting/permutations:

A = [1.0 1 1;
     2   4 8;
     1   4 9]

L,U = lu(A, NoPivot()) # No Pivot is needed to tell lu not to using permutations

# This matches what we derived by hand and indeed:

@test A ≈ L*U

# We can use an LU factorisation to reduce solving a linear system to inverting triangular matrices:

b = randn(3)
c = L \ b # computed using forward elimination (even though L is a Matrix, \ detects it is lower triangular)
x = U \ c # computed using back substitution
@test A \ b == x # matches Julia's \ exactly, since this matrix did not need pivoting

# If a matrix has a zero on a pivot we know by equivalence to Gaussian elimination that an LU factorisation
# does not exist:

A[1,1] = 0
lu(A, NoPivot()) # throws an error

# But even if it has a very small but non-zero entry we get huge errors:

A[1,1] = 1E-14
L,U = lu(A, NoPivot()) # Succeeds but suddenly U is on order of 2E14!

norm(A \ b - U\(L\b)) # Very large error! A \ b uses pivoting now.

# **WARNING** The parantheses are important: algebra is left-associatitive so had we written `U\L\b` this would have been interpreted as
# `(L\U) \ b` which would have meant `inv(inv(L)*U)*b == U \ (L*b).


# **Problem 1** For `A` defined above, consider setting the `A[1,1] = ε` for `ε = 10.0 ^ (-k)` for `k = 0,…,14`
# with the right-hand side `b = [1,2,3]`.
# Use a scale the $y$-axis logarithmically to conjecture the growth rate in the error of using LU compared to `\` as $k → ∞$. 
# Hint: you can either allocate a vector of errors that is populated in a for-loop or write a simple comprehension.

## TODO: Do a log-log plot for A with its 1,1 entry set to different ε and guess the growth rate.
## SOLUTION

A = [1.0 1 1;
     2   4 8;
     1   4 9]

b = [1,2,3]


n = 15
errs = zeros(n)
for k = 1:n
    A[1,1] = 10.0 ^ (1-k)
    L,U = lu(A, NoPivot())
    errs[k] = norm(A\b - L \ (U \ b))
end

scatter(0:n-1, errs; yscale=:log10, yticks = 10.0 .^ (0:30), xticks = 1:15)
## The error grows exponentially, like 10^(2k)

## END


# **Problem 2(a)** Consider the Helmholtz equations
# $$
# \begin{align*}
# u(0) &= 0 \\
# u(1) &= 0 \\
# u'' + k^2 u &= {\rm e}^x
# \end{align*}
# $$
# discretised with finite-differences to result in a tridiagonal system.
# Use the `lu` function without pivoting to
# compute the LU factorization of the tridiagonal matrix. What sparsity structure
# do you observe in `L` and `U`? Does this structure depend on $n$ or $k$?

## TODO: Apply lu to the discretisation for Helmholtz derived in the last lab and investigate its structure.

## SOLUTION


## We make a function that returns the Helmholtz matrix:
function helmholtz(n, k)
    x = range(0, 1; length = n + 1)
    h = step(x)
    Tridiagonal([fill(1/h^2, n-1); 0], 
                    [1; fill(k^2-2/h^2, n-1); 1], 
                    [0; fill(1/h^2, n-1)])
end

lu(helmholtz(20, 2), NoPivot()) # L is lower bidiagonal and U is lower bidiagonal, regardless of n or k
## END


# ## III.3.2 PLU Factorisation

# In general it is necessary to do pivoting, a feature you have seen
# in Gaussian elimination but as Problem 1 demonstrates we need to do so even if we do not encounter
# a zero. This corresponds to a factorisation of the form
# $$
#  A = P^⊤LU
# $$
# where $P$ is a permutation matrix, $L$ is lower triangular and $U$ is upper triangular.
# We compute this as follows, printing out the permutation:

A = [0.1 1 1;
     2   4 8;
     1   4 9]

L,U,σ = lu(A)
σ

# The permutation is encoded as a vector $σ$. More precisely, we have
# $$
#     P^⊤ 𝐯 = 𝐯[σ]
# $$
# Thus we can solve a linear system as follows: we first permute the entries of the right-hand side:

b = [10,11,12]
b̃ = b[σ] # permute the entries to [b[2],b[3],b[1]]

# We now solve as before:
c = L \ b̃ # invert L with forward elimination
x = U \ c # invert U with back substitution

@test x == A \ b # \ also use PLU to do the solve so these exactly equal

# **Problem 2(b)** Repeat Problem 2(a) but with a PLU decomposition. 
# Are $L$ and $U$ still banded?

## TODO:

## SOLUTION
lu(helmholtz(20, 2)).L # L is no longer banded: its penultimate row is dense
## END

# ## III.3.3 Cholesky Factorisation




# 4. Timings and Stability

The different factorisations have trade-offs between speed and stability.
First we compare the speed of the different factorisations on a symmetric positive
definite matrix, from fastest to slowest:

```julia
n = 100
A = Symmetric(rand(n,n)) + 100I # shift by 10 ensures positivity
@btime cholesky(A);
@btime lu(A);
@btime qr(A);
```
On my machine, `cholesky` is ~1.5x faster than `lu`,  
which is ~2x faster than QR. 



In terms of stability, QR computed with Householder reflections
(and Cholesky for positive definite matrices) are stable, 
whereas LU is usually unstable (unless the matrix
is diagonally dominant). PLU is a very complicated story: in theory it is unstable,
but the set of matrices for which it is unstable is extremely small, so small one does not
normally run into them.

Here is an example matrix that is in this set. 
```julia
function badmatrix(n)
    A = Matrix(1I, n, n)
    A[:,end] .= 1
    for j = 1:n-1
        A[j+1:end,j] .= -1
    end
    A
end
A =1badmatrix(5)
```
Note that pivoting will not occur (we do not pivot as the entries below the diagonal are the same magnitude as the diagonal), thus the PLU Factorisation is equivalent to an LU factorisation:
```julia
L,U = lu(A)
```
But here we see an issue: the last column of `U` is growing exponentially fast! Thus when `n` is large
we get very large errors:
```julia
n = 100
b = randn(n)
A = badmatrix(n)
norm(A\b - qr(A)\b) # A \ b still uses lu
```
Note `qr` is completely fine:
```julia
norm(qr(A)\b - qr(big.(A)) \b) # roughly machine precision
```

Amazingly, PLU is fine if applied to a small perturbation of `A`:
```julia
ε = 0.000001
Aε = A .+ ε .* randn.()
norm(Aε \ b - qr(Aε) \ b) # Now it matches!
```



The big _open problem_ in numerical linear algebra is to prove that the set of matrices
for which PLU fails has extremely small measure.




    Note hidden in this proof is a simple algorithm form computing the Cholesky factorisation.

**Algorithm 3 (Cholesky)** 
```julia
function mycholesky(A)
    T = eltype(A)
    n,m = size(A)
    if n ≠ m
        error("Matrix must be square")
    end
    if A ≠ A'
        error("Matrix must be symmetric")
    end
    T = eltype(A)
    L = LowerTriangular(zeros(T,n,n))
    Aⱼ = copy(A)
    for j = 1:n
        α,𝐯 = Aⱼ[1,1],Aⱼ[2:end,1]
        if α ≤ 0
            error("Matrix is not SPD")
        end 
        L[j,j] = sqrt(α)
        L[j+1:end,j] = 𝐯/sqrt(α)

        # induction part
        Aⱼ = Aⱼ[2:end,2:end] - 𝐯*𝐯'/α
    end
    L
end

A = Symmetric(rand(100,100) + 100I)
L = mycholesky(A)
@test A ≈ L*L'
```


This algorithm succeeds if and only if $A$ is symmetric positive definite.



# We now consider a Cholesky factorisation for tridiagonal matrices. Since we are assuming the
# matrix is symmetric, we will use a special type `SymTridiagonal` that captures the symmetry.
# In particular, `SymTridiagonal(dv, eu) == Tridiagonal(ev, dv, ev)`.

# **Problem 3** Complete the following
# implementation of `mycholesky` to return a `Bidiagonal` cholesky factor in $O(n)$ operations.


## return a Bidiagonal L such that L'L == A (up to machine precision)
## You are allowed to change A
function mycholesky(A::SymTridiagonal)
    d = A.dv # diagonal entries of A
    u = A.ev # sub/super-diagonal entries of A
    T = float(eltype(A)) # return type, make float in case A has Ints
    n = length(d)
    ld = zeros(T, n) # diagonal entries of L
    ll = zeros(T, n-1) # sub-diagonal entries of L

    ## TODO: populate the diagonal entries ld and the sub-diagonal entries ll
    ## of L so that L*L' ≈ A
    ## SOLUTION
    ld[1] = sqrt(d[1])
    for k = 1:n-1
        ll[k] = u[k]/ld[k]
        ld[k+1] = sqrt(d[k+1]-ll[k]^2)
    end
    ## END

    Bidiagonal(ld, ll, :L)
end

n = 1000
A = SymTridiagonal(2*ones(n),-ones(n-1))
L = mycholesky(A)
@test L*L' ≈ A


# ## III.4 Polynomial Interpolation and Regression

# ### III.4.1 Polynomial Interpolation

# Thus a quick-and-dirty way to to do interpolation is to invert the Vandermonde matrix
# (which we saw in the least squares setting with more samples then coefficients):

using Plots, LinearAlgebra
f = x -> cos(10x)
n = 5

x = range(0, 1; length=n+1)# evenly spaced points (BAD for interpolation)
V = x .^ (0:n)' # Vandermonde matrix
c = V \ f.(x) # coefficients of interpolatory polynomial
p = x -> dot(c, x .^ (0:n))

g = range(0,1; length=1000) # plotting grid
plot(g, f.(g); label="function")
plot!(g, p.(g); label="interpolation")
scatter!(x, f.(x); label="samples")



# When $m = n$ a least squares fit by a polynomial becomes _interpolation_:
# the approximating polynomial will fit the data exactly. That is, for
# $$
# p(x) = ∑_{k = 0}^n p_k x^k
# $$
# and $x_1, …, x_n ∈ ℝ$, we choose $p_k$ so that $p(x_j) = f(x_j)$ for
# $j = 1, …, n$. 

# **Problem 1.1** Complete the following function which returns a rectangular _Vandermonde matrix_:
# a matrix $V ∈ ℝ^{m × n}$ such that
# $$
# V * \begin{bmatrix} p_0\\ ⋮ \\p_n \end{bmatrix} = \begin{bmatrix} p(x_1)\\ ⋮ \\p(x_m) \end{bmatrix}
# $$

function vandermonde(𝐱, n) # 𝐱 = [x_1,…,x_m]
    m = length(𝐱)
    ## TODO: Make V
    ## SOLUTION
    ## There are also solutions using broadcasting or for loops.
    ## e.g. 
    ## 𝐱 .^ (0:n-1)'
    [𝐱[j]^k for j = 1:m, k = 0:n-1]
    ## END
end

n = 1000
𝐱 = range(0, 0.5; length=n)
V = vandermonde(𝐱, n) # square Vandermonde matrix
## if all coefficients are 1 then p(x) = (1-x^n)/(1-x)
@test V * ones(n) ≈ (1 .- 𝐱 .^ n) ./ (1 .- 𝐱)


# Inverting the square Vandermonde matrix is a way of computing coefficients from function
# samples. That is, solving
# $$
# V𝐜 = \begin{bmatrix} f(x_1) \\ ⋮ \\ f(x_n) \end{bmatrix}
# $$
# Gives the coefficients of a polynomial $p(x)$ so that $p(x_j) = f(x_j)$.
# Whether an interpolation is actually close to a function is a subtle question,
# involving properties of the function, distribution of the sample points $x_1,…,x_n$,
# and round-off error.
# A classic example is:
# $$
#   f_M(x) = {1 \over M x^2 + 1}
# $$
# where the choice of $M$ can dictate whether interpolation at evenly spaced points converges.

# **Problem 1.2** Interpolate $1/(4x^2+1)$ and $1/(25x^2 + 1)$ at an evenly spaced grid of $n$
# points, plotting the solution at a grid of $1000$ points. For $n = 50$ does your interpolation match
# the true function?  Does increasing $n$ to 400 improve the accuracy? How about using `BigFloat`?

n = 50
𝐱 = range(-1, 1; length=n)
𝐠 = range(-1, 1; length=1000) # plotting grid

## TODO: interpolate 1/(10x^2 + 1) and 1/(25x^2 + 1) at $𝐱$, plotting both solutions evaluated at
## the grid 𝐠. Hint: use a rectangular Vandermonde matrix to evaluate your polynomial on 𝐠. Remember
## `plot(𝐱, 𝐟)` will create a new plot whilst `plot!(𝐱, 𝐟)` will add to an existing plot.

## SOLUTION
V = vandermonde(𝐱, n)
V_g = vandermonde(𝐠, n)
f_4 = x -> 1/(4x^2 + 1)
𝐜_4 = V \ f_4.(𝐱)
f_25 = x -> 1/(25x^2 + 1)
𝐜_25 = V \ f_25.(𝐱)

plot(𝐠, V_g*𝐜_4; ylims=(-1,1))
plot!(𝐠, V_g*𝐜_25)
## We see large errors near ±1 for both examples. 
## END
#
## TODO: repeat the experiment with `n = 400` and observe what has changed.
## SOLUTION
n = 400
𝐱 = range(-1, 1; length=n)
𝐠 = range(-1, 1; length=1000) # plotting grid

V = vandermonde(𝐱, n)
V_g = vandermonde(𝐠, n)
f_4 = x -> 1/(4x^2 + 1)
𝐜_4 = V \ f_4.(𝐱)
f_25 = x -> 1/(25x^2 + 1)
𝐜_25 = V \ f_25.(𝐱)

plot(𝐠, V_g*𝐜_4; ylims=(-1,1))
plot!(𝐠, V_g*𝐜_25)
## Still does not converge
## END
#
## TODO: repeat the experiment with `n = 400` and using `BigFloat` and observe what has changed.
## Hint: make sure to make your `range` be `BigFloat` valued, e.g., by using `big`.
## SOLUTION
n = 400
𝐱 = range(big(-1), 1; length=n)
𝐠 = range(big(-1), 1; length=1000) # plotting grid

V = vandermonde(𝐱, n)
V_g = vandermonde(𝐠, n)
f_4 = x -> 1/(4x^2 + 1)
𝐜_4 = V \ f_4.(𝐱)
f_25 = x -> 1/(25x^2 + 1)
𝐜_25 = V \ f_25.(𝐱)

plot(𝐠, V_g*𝐜_4; ylims=(-1,1))
plot!(𝐠, V_g*𝐜_25)
## With M = 4 it looks like it now is converging. This suggests the issue before was numerical error.
## For M = 25 the solution is even less accurate, which suggests the issue is a lack of mathematical
## convergence.

## END


**Example 1 (quadratic fit)** Suppose we want to fit noisy data by a quadratic
$$
p(x) = p₀ + p₁ x + p₂ x^2
$$
That is, we want to choose $p₀,p₁,p₂$ at data samples $x_1, …, x_m$ so that the following is true:
$$
p₀ + p₁ x_k + p₂ x_k^2 ≈ f_k
$$
where $f_k$ are given by data. We can reinterpret this as a least squares problem: minimise the norm
$$
\left\| \begin{bmatrix} 1 & x_1 & x_1^2 \\ ⋮ & ⋮ & ⋮ \\ 1 & x_m & x_m^2 \end{bmatrix}
\begin{bmatrix} p₀ \\ p₁ \\ p₂ \end{bmatrix} - \begin{bmatrix} f_1 \\ ⋮ \\ f_m \end{bmatrix} \right \|
$$
We can solve this using the QR decomposition:
```julia
m,n = 100,3

x = range(0,1; length=m) # 100 points
f = 2 .+ x .+ 2x.^2 .+ 0.1 .* randn.() # Noisy quadratic

A = x .^ (0:2)'  # 100 x 3 matrix, equivalent to [ones(m) x x.^2]
Q,R̂ = qr(A)
Q̂ = Q[:,1:n] # Q represents full orthogonal matrix so we take first 3 columns

p₀,p₁,p₂ = R̂ \ Q̂'f
```
We can visualise the fit:
```julia
p = x -> p₀ + p₁*x + p₂*x^2

scatter(x, f; label="samples", legend=:bottomright)
plot!(x, p.(x); label="quadratic")
```
Note that `\` with a rectangular system does least squares by default:
```julia
A \ f
```



# **Problem 1.3** Repeat the previous problem but now using _least squares_: instead of interpolating,
# use least squares on a large grid: choose the coefficients of a degree $(n-1)$ polynomial so that
# $$
#     \left\| \begin{bmatrix} p(x_1) \\ ⋮ \\ p(x_m) \end{bmatrix} - \begin{bmatrix} f(x_1) \\ ⋮ \\ f(x_m) \end{bmatrix} \right \|.
# $$
# is minimised.
# Does this improve the convergence properties? Do you think convergence for a least squares approximation
# is dictated by the radius of convergence of the corresponding Taylor series?
# Hint: use the rectangular Vandermonde matrix to setup the Least squares system.

n = 50 # use basis [1,x,…,x^(49)]
𝐱 = range(-1, 1; length=500) # least squares grid
𝐠 = range(-1, 1; length=2000) # plotting grid

## TODO: interpolate 1/(10x^2 + 1) and 1/(25x^2 + 1) at $𝐱$, plotting both solutions evaluated at
## the grid 𝐠. Hint: use a rectangular Vandermonde matrix to evaluate your polynomial on 𝐠. Remember
## `plot(𝐱, 𝐟)` will create a new plot whilst `plot!(𝐱, 𝐟)` will add to an existing plot.

## SOLUTION
V = vandermonde(𝐱, n)
V_g = vandermonde(𝐠, n)
f_4 = x -> 1/(4x^2 + 1)
𝐜_4 = V \ f_4.(𝐱)
f_25 = x -> 1/(25x^2 + 1)
𝐜_25 = V \ f_25.(𝐱)

plot(𝐠, V_g*𝐜_4; ylims=(-1,1))
plot!(𝐠, V_g*𝐜_25)

## Yes, now both approximations appear to be converging.
## This is despite the radius of convergence of both functions being
## smaller than the interval of interpolation.

## END
