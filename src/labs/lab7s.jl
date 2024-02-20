# # MATH50003 (2023–24)
# # Lab 7: III.5 Orthogonal and Unitary Matrices and III.6 QR Factorisation

# This lab explores orthogonal matrices, including rotations and reflections.
# We will construct special types to capture the structure of these orthogonal operations,
# With the goal of implementing fast matrix*vector and matrix\vector operations.
# We also compute the QR factorisation with Householder reflections.

# **Learning Outcomes**
#
# Mathematical knowledge:
#
# 1. 

using LinearAlgebra, Test
import Base: getindex, *, size, \



# ## III.5 Orthogonal and Unitary Matrices

# ### III.5.2 Reflections


function dense_householderreflection(x)
    y = copy(x)
    if x[1] == 0
        y[1] += norm(x) 
    else # note sign(z) = exp(im*angle(z)) where `angle` is the argument of a complex number
        y[1] += sign(x[1])*norm(x) 
    end
    w = y/norm(y)
    I - 2*w*w'
end


# **Problem 1(a)** Complete the implementation of a type representing an n × n
# reflection that supports `Q[k,j]` in $O(1)$ operations and `*` in $O(n)$ operations.
# The reflection may be complex (that is, $Q ∈ U(n)$ is unitary).

## Represents I - 2v*v'
struct Reflection <: AbstractMatrix{ComplexF64}
    v::Vector{ComplexF64} # We are assuming v is normalised. 
end

size(Q::Reflection) = (length(Q.v),length(Q.v))

## getindex(Q, k, j) is synonym for Q[k,j]
function getindex(Q::Reflection, k::Int, j::Int)
    ## TODO: implement Q[k,j] == (I - 2v*v')[k,j] but using O(1) operations.
    ## Hint: the function `conj` gives the complex-conjugate
    ## SOLUTION
    if k == j
        1 - 2Q.v[k]*conj(Q.v[j])
    else
        - 2Q.v[k]*conj(Q.v[j])
    end
    ## END
end
function *(Q::Reflection, x::AbstractVector)
    ## TODO: implement Q*x, equivalent to (I - 2v*v')*x but using only O(n) operations
    ## SOLUTION
    x - 2*Q.v * dot(Q.v,x) # (Q.v'*x) also works instead of dot
    ## END
end

## If your code is correct, these "unit tests" will succeed
n = 10
x = randn(n) + im*randn(n)
Q = Reflection(x)
v = x/norm(x)
@test Q == I-2v*v'
@test Q'Q ≈ I
n = 100_000
x = randn(n) + im*randn(n)
v = x/norm(x)
Q = Reflection(v)
@test Q*v ≈ -v




# **Problem 1(b)** Complete the following implementation of a Housholder reflection  so that the
# unit tests pass, using the `Reflection` type created above.
# Here `s == true` means the Householder reflection is sent to the positive axis and `s == false` is the negative axis.

function householderreflection(s::Bool, x::AbstractVector)
    ## TODO: return a Reflection corresponding to a Householder reflection
    ## SOLUTION
    y = copy(x) # don't modify x
    if s
        y[1] -= norm(x)
    else
        y[1] += norm(x)
    end
    Reflection(y)
    ## END
end

x = randn(5)
Q = householderreflection(true, x)
@test Q isa Reflection
@test Q*x ≈ [norm(x);zeros(eltype(x),length(x)-1)]

Q = householderreflection(false, x)
@test Q isa Reflection
@test Q*x ≈ [-norm(x);zeros(eltype(x),length(x)-1)]


# ---------

# **Problem 2**
# Complete the definition of `Reflections` which supports a sequence of reflections,
# that is,
# $$
# Q = Q_{𝐯_1} ⋯ Q_{𝐯_m}
# $$
# where the vectors are stored as a matrix $V ∈ ℂ^{n × m}$ whose $j$-th column is $𝐯_j∈ ℂ^n$, and
# $$
# Q_{𝐯_j} = I - 2 𝐯_j 𝐯_j^⋆
# $$
# is a reflection.


struct Reflections <: AbstractMatrix{ComplexF64}
    V::Matrix{ComplexF64} # Columns of V are the householder vectors
end

size(Q::Reflections) = (size(Q.V,1), size(Q.V,1))


function *(Q::Reflections, x::AbstractVector)
    ## TODO: Apply Q in O(mn) operations by applying
    ## the reflection corresponding to each column of Q.V to x
    
    ## SOLUTION
    m,n = size(Q.V)
    for j = n:-1:1
        x = Reflection(Q.V[:, j]) * x
    end
    ## END

    x
end

function getindex(Q::Reflections, k::Int, j::Int)
    ## TODO: Return Q[k,j] in O(mn) operations (hint: use *)

    ## SOLUTION
    T = eltype(Q.V)
    m,n = size(Q)
    eⱼ = zeros(T, m)
    eⱼ[j] = one(T)
    return (Q*eⱼ)[k]
    ## END
end

Y = randn(5,3)
V = Y * Diagonal([1/norm(Y[:,j]) for j=1:3])
Q = Reflections(V)
@test Q ≈ (I - 2V[:,1]*V[:,1]')*(I - 2V[:,2]*V[:,2]')*(I - 2V[:,3]*V[:,3]')
@test Q'Q ≈ I




# III.6 QR Factorisation

# This proof by induction leads naturally to an iterative algorithm. Note that $\tilde Q$ is a product of all
# Householder reflections that come afterwards, that is, we can think of $Q$ as:
# $$
# Q = Q_1 \tilde Q_2 \tilde Q_3 ⋯ \tilde Q_n\qquad\hbox{for}\qquad \tilde Q_j = \begin{bmatrix} I_{j-1} \\ & Q_j \end{bmatrix}
# $$
# where $Q_j$ is a single Householder reflection corresponding to the first column of $A_j$. 
# This is stated cleanly in Julia code:




function householderqr(A)
    T = eltype(A)
    m,n = size(A)
    if n > m
        error("More columns than rows is not supported")
    end

    R = zeros(T, m, n)
    Q = Matrix(one(T)*I, m, m)
    Aⱼ = copy(A)

    for j = 1:n
        𝐚₁ = Aⱼ[:,1] # first columns of Aⱼ
        Q₁ = dense_householderreflection(𝐚₁)
        Q₁Aⱼ = Q₁*Aⱼ
        α,𝐰 = Q₁Aⱼ[1,1],Q₁Aⱼ[1,2:end]
        Aⱼ₊₁ = Q₁Aⱼ[2:end,2:end]

        # populate returned data
        R[j,j] = α
        R[j,j+1:end] = 𝐰

        # following is equivalent to Q = Q*[I 0 ; 0 Qⱼ]
        Q[:,j:end] = Q[:,j:end]*Q₁

        Aⱼ = Aⱼ₊₁ # this is the "induction"
    end
    Q,R
end

m,n = 100,50
A = randn(m,n)
Q,R = householderqr(A)
@test Q'Q ≈ I
@test Q*R ≈ A


# Note because we are forming a full matrix representation of each Householder
# reflection this is a slow algorithm, taking $O(n^4)$ operations. The problem sheet
# will consider a better implementation that takes $O(n^3)$ operations.





# In lectures we did a quick-and-dirty implementation of Householder QR.
# One major issue though: it used $O(m^2 n^2)$ operations, which is too many!
# By being more careful about how we apply and store reflections we can avoid this,
# in particular, taking advantage of the types `Reflection` and `Reflections` we developed
# last lab. 

# **Problem 2** Complete the following function that implements
# Householder QR for a real matrix $A ∈ ℝ^{m × n}$ where $m ≥ n$ using only $O(mn^2)$ operations, using 
#  `Reflection` and `Reflections`.
# Hint: We have added the overload functions `*(::Reflection, ::AbstractMatrix)` and
# `*(::Reflections, ::AbstractMatrix)` so that they can be multiplied by an $m × n$ matrix in $O(mn)$ operations.


function householderqr(A)
    T = eltype(A)
    m,n = size(A)
    if n > m
        error("More columns than rows is not supported")
    end

    R = zeros(T, m, n)
    Q = Reflections(zeros(T, m, n))
    Aⱼ = copy(A)

    for j = 1:n
        ## TODO: rewrite householder QR to use Reflection and
        ## Reflections, in a way that one achieves O(mn^2) operations
        ## SOLUTION
        𝐚₁ = Aⱼ[:,1] # first columns of Aⱼ
        Q₁ = householderreflection(𝐚₁[1] < 0, 𝐚₁)
        Q₁Aⱼ = Q₁*Aⱼ
        α,𝐰 = Q₁Aⱼ[1,1],Q₁Aⱼ[1,2:end]
        Aⱼ₊₁ = Q₁Aⱼ[2:end,2:end]

        ## populate returned data
        R[j,j] = α
        R[j,j+1:end] = 𝐰

        Q.V[j:end, j] = Q₁.v

        Aⱼ = Aⱼ₊₁ # this is the "induction"
        ## END
    end
    Q,R
end

A = randn(600,400)
Q,R = householderqr(A)
@test Q*R ≈ A

