# # MATH50003 Numerical Analysis (2023–2024) Mock Computer-based Exam

# Instructions for uploading and downloading:

# 1. Rename the file to include your CID.
# 2. You have 15 mins to download the exam beginning at 16:00 on 7 March. **DO NOT** write or type anything during this time.
# 2. You have 1 hour to complete the exam beginning at 16:15 on 7 March. You **MUST STOP** typing at 17:15.
# 3. Deadline is 17:30 on 7 March to upload the completed Jupyter notebook (`.ipynb`) to Blackboard. 
# 5. Once uploaded, re-download the file before the final submission time to confirm it is correct.
# You are allowed to upload additional submissions but only the last valid upload before 17:30 will be used.
# 6. If uploading via Blackboard fails you may e-mail the UG Office: maths.exams@imperial.ac.uk

# Instructions for the exam:

# 1. For each problem, replace the `# TODO` to complete the question.
# The unit tests are provided to help you test your answers, but do not guarantee that the answer is correct.
# 2. Problems are marked A/B/C to indicate difficulty ("A" being most difficult).
# 3. All questions are worth 10 marks.Partial credit will be awarded for reasonable attempts or comments outlining a solution even if the tests
# are not passed.
# 3. If you have technical queries please email s.olver@imperial.ac.uk. Any other queries
# should be sent to the UG Office: maths.exams@imperial.ac.uk
# 4. You may use existing code from anywhere
# but you are **REQUIRED** to cite the source if it is not part of the module material,
# by including a weblink in a comment. 
# 5. You **MUST NOT** ask for help online or
# communicate with others within or outside the module.
# Failure to follow these rules will be considered misconduct.
# 6. **NO USAGE of AI tools** such as ChatGPT or GitHub Co-Pilot.
# 7. **TURN OFF notifications** on your machine. In particular, close background processes for any chat applications (iMessage, WhatsApp, WeChat).


# You should use the following packages:

using LinearAlgebra, SetRounding, Test

# **WARNING** It may be necessary to restart the kernel if issues arise. Remember to reload the packages
# when you do so.

# -----

# **Problem 1 (C)** Use second-order divided differences
# with an appropriately chosen $h$ to approximate the second derivative of
# $$
# f(x) = \cos(x^2)
# $$
# at $x = 0.1$ with an error less than $10^{-5}$. Note you are not required to choose a "quasi-optimal"
# value for $h$, as long as your choice achieves the specified accuracy.

function fd2(x)
    ## TODO: implement a second-order finite-difference rule 
    ## to approximate f''(x)
    ## for f(x) = cos(x^2)
    ## with step-size h chosen to get sufficient accuracy
    
end


@test abs(fd2(0.1) + 2*sin(0.1^2) + 4*0.1^2*cos(0.1^2)) ≤ 1E-5

# **Problem 2 (B)** Implement powers of dual numbers to a float $(a+bϵ)^c$ and
# to a dual number $(a+bϵ)^{c+dϵ}$, in a way that is consistent with a "dual-extension",
# e.g. if $f(x) = x^{3/2}$ or $f(x) = x^x$ then we want to define the power function so that
# in both cases $f(a + bϵ) = f(a) + bf'(a)ϵ$.
# Hint: for the second part recall $x^y = \exp(y \log x)$ which reduces the problem
# to single-argument functions where the "dual-extension" is easy to define.

## Represents a + b*ε
struct Dual
    a
    b
end

import Base: ^, *, isapprox
*(x::Dual, y::Dual) = Dual(x.a*y.a, x.a*y.b + x.b*y.a)
isapprox(x::Dual, y::Dual) = x.a ≈ y.a && x.b ≈ y.b # used in tests

function ^(x::Dual, c::Real)
    ## TODO: Implement Dual(a,b)^c returning a Dual whose b-component is consistent
    ## with differentiation.
    
end

@test Dual(1.0,2.0)^0.0 == Dual(1.0, 0.0)
@test Dual(1.0,2.0)^0.5 == Dual(1.0, 1.0)
@test Dual(1.0,2.0)^(-0.5) == Dual(1.0, -1.0)

function ^(x::Dual, y::Dual)
    ## TODO: Implement Dual(a,b)^Dual(c,d), returning a Dual in a way that is consistent with
    ## differentiation: i.e. for the function f(x) = x^x, f(Dual(2,1)) should return
    ## Dual(f(2), f′(2)) where f′(x) denotes the derivative of f.
    
end


@test Dual(2.0, 1.0) ^ Dual(3.0, 1.0) ≈ Dual(8.0,8*(3/2 + log(2)))

# **Problem 3 (A)** Consider a 2nd order version of a dual number:
# $$
# a + b ϵ_1 + c ϵ_2
# $$
# such that
# $$
# \begin{align*}
# ϵ_1^2 &= ϵ_2, \\
# ϵ_2^2 &= ϵ_1 ϵ_2 =  0.
# \end{align*}
# $$
# Complete the following implementation supporting `+` and `*`. You may
# assume `a,b,c` are `Float64`. Hint: you may need to work out on paper
# how to multiply `(a + b*ϵ_1 + c*ϵ_2)*(d + e*ϵ_1 + f*ϵ_2)` using the
# relationship above.

import Base: *, +, ^
struct Dual2
    a
    b
    c
end

function +(s::Dual2, t::Dual2)
    ## TODO: Implement Dual2(...) + Dual2(...), returning a Dual2
    
end

function +(s::Dual2, c::Real)
    ## TODO: Implement Dual2(...) + c, returning a Dual2
    
end

function *(c::Number, s::Dual2)
    ## TODO: Implement c * Dual2(...), returning a Dual2
    
end

function *(s::Dual2, t::Dual2)
    ## TODO: Implement Dual2(...) * Dual2(...), returning a Dual2

    
end

f = x -> x*x*x + 2x + 1
x = 0.1
@test f(Dual2(x,1.,0.)) == Dual2(f(x), 3x^2+2, 6x / 2)

## This has computed the first and second derivatives as
## f(x) + f'(x)*ϵ_1 + f''(x)/2*ϵ_2 == (x^3 + x) + (3x^2+1)*ϵ_1 + 6x/2*ϵ_2

# **Problem 4 (C)**  Implement the function `issub` that determines whether a `Float16` is a sub-normal number.
# DO NOT use the inbuilt routine `issubnormal`. 


function issub(x::Float16)
    ## TODO: return true if x is a sub-normal float. Otherwise return false
    
end

@test issub(Float16(0))
@test issub(nextfloat(Float16(0)))
@test issub(prevfloat(Float16(0)))
@test !issub(Float16(1))
@test !issub(reinterpret(Float16,0b0000010000000000))
@test issub(reinterpret(Float16,0b0000001111111111))


# **Problem 5 (C)** Complete the following function `divideby3(x)` that
# returns a tuple `a,b` such that `a` is the largest `Float64` less 
# than or equal to `x/3` and `b` is the smallest `Float64` greater than or equal to `x/3`,
# using the `setrounding` function. You may assume the input is a `Float64`.

function divideby3(x)
    ## TODO: return a tuple a,b so that a ≤ x ≤ b where b is either equal to a or the next float
    
end

x = 0.1 # arbitary x
a,b = divideby3(x)
@test a ≤ big(x)/3 ≤ b
@test b == nextfloat(a)



# **Problem 6 (B)** Complete the implementation of `LowerTridiagonal` which represents an $n × n$ banded matrix with
# bandwidths $(l,u) = (2,0)$ by storing only its diagonal, sub-diagonal, and second-sub-diagonal as vectors.
# Overload `getindex`, as well as `*` to multiply a lower tridiagonal matrix times a vector in $O(n)$ operations. 

import Base: getindex,  size, *

struct LowerTridiagonal <: AbstractMatrix{Float64}
    d::Vector{Float64}   # diagonal entries of length n
    dl::Vector{Float64}  # sub-diagonal entries of length n-1
    dl2::Vector{Float64} # second-sub-diagonal entries of length n-2
end

size(L::LowerTridiagonal) = (length(L.d),length(L.d))

function getindex(L::LowerTridiagonal, k::Int, j::Int)
    d, dl, dl2 = L.d, L.dl, L.dl2
    ## TODO: return L[k,j].
    
end


function *(L::LowerTridiagonal, x::AbstractVector)
    ## TODO: Return L*x but computed in O(n) operations
    
end

n = 10
d, dl, dl2 = randn(n), randn(n-1), randn(n-2)
L = LowerTridiagonal(d, dl, dl2)
@test L == diagm(0 => d, -1 => dl, -2 => dl2)
x = randn(n)
@test L*x ≈ diagm(0 => d, -1 => dl, -2 => dl2)*x


# **Problem 7 (B)** Complete the function `airy(n)` that returns a length-$n+1$ vector
# $$
# \begin{bmatrix}
# u_0 \\
# ⋮ \\
# u_n
# \end{bmatrix}
# $$
# such that $u_k ≈ u(x_k)$ approximates the solution to the equation
# $$
# \begin{align*}
# u(0) &= 1 \\
# u(1) &= 0 \\
# u'' - x u &= 0
# \end{align*}
# $$
# at the point $x_k = k/n$ using the second order finite-difference approximation:
# $$
# u''(x_k) ≈ {u_{k-1} - 2u_k + u_{k+1} \over h^2}
# $$
# for $k = 1, …, n-1$. The solution must use only $O(n)$ operations.


function airy(n)
    ## TODO: return a Vector{Float64} approximating the solution to the ODE
    
end

u = airy(999)
@test length(u) == 1000
@test u[1] == 1
@test u[end] == 0
## this compares agianst the exact formula
@test abs(u[500] - 0.4757167332829094) ≤ 2E-8



# **Problem 8 (C)** Approximate $\exp x$ by a cubic polynomial by minimising
# the least squares error when sampled at $n$ evenly spaced points in $[0,1]$,
# that is, $x_k = (k-1)/(n-1)$ for $k = 1,…,n$,
# returning the coefficients in the monomial basis.


function expfit(n)
    ## TODO: return the coefficients [c_0,c_1,c_2,c_3] of the polynomial
    ## c_0 + c_1*x + c_2*x^2 + c_3*x^3 that minimises the L^2 error at n
    ## evenly spaced samples
    
end

c = expfit(1000)
x = 0.1
@test abs(c[1] + c[2]*x + c[3]*x^2 + c[4]*x^3 - exp(x)) ≤ 1E-3


# **Problem 9.1 (B)** Implement `lowerhouseholderreflection(x)` where `x` is a vector representing $𝐱 ∈ ℝ^n$
#  to return a reflection $Q$  satisfying $Q 𝐱 = -\| 𝐱 \| 𝐞_n$.
#  The function `lowerhouseholderreflection(x)` should return a `Matrix{Float64}`.
# You may assume that `x` is a `Vector{Float64}`.


function lowerhouseholderreflection(x)
    ## TODO: implement the householder reflector defined above
    
end
x = [1.0,2,3,4]
Q = lowerhouseholderreflection(x)
@test Q*x ≈ [zeros(3); -norm(x)]
@test Q'Q ≈ I
@test Q ≈ Q'

# **Problem 9.2 (A)** Complete the function `ql(A)` that
# returns a QL decomposition, that is, `Q` is an orthogonal
# matrix and `L` is lower triangular satisfying (up to rounding) `A == Q*L`. You may assume that `A`
# is a square `Matrix{Float64}`. Hint: use the previous part to lower triangularise.

function ql(A)
    m,n = size(A)
    m == n || error("not square")
    ## TODO Create Q and L such that Q'Q == I, L is lower triangular, and Q*L ≈ A
    
end

A = [1.0 2 3; 1 4 9; 1 1 1]
Q,L = ql(A)
@test Q'Q ≈ I
@test Q*L ≈ A
@test L ≈ tril(L) # it is acceptable to have small non-zero entries in L

