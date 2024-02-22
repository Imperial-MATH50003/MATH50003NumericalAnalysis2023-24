# # MATH50003 (2023–24)
# # Lab 7: III.5 Orthogonal and Unitary Matrices and III.6 QR Factorisation

# This lab explores orthogonal matrices, including rotations and reflections.
# We will construct special types to capture the structure of these orthogonal operations,
# With the goal of implementing fast matrix*vector and matrix\vector operations.
# We also compute the QR factorisation with Householder reflections, and use this
# to solve least squares problems.

# **Learning Outcomes**
#
# Mathematical knowledge:
#
# 1. Constructing rotation and reflection matrices.
# 2. Computing the QR factorisation using reflections.
# 3. Computing a tridiagonal QR factorisation using rotations.
# 4. The relationship between QR and least squares.
#
# Coding knowledge:
#
# 1. The `atan(y,x)` function.
# 2. Templating fields in a type.
# 3. Using the `qr` function to solve least squares.



using LinearAlgebra, Test



# ## III.5 Orthogonal and Unitary Matrices

# Here we explore representing rotations and reflections, which are
# special types of orthogonal/unitary matrices. 

# ### III.5.1 Rotations

# A rotation matrix has the form
# $$
#  \begin{bmatrix} c & -s \\ s & c \end{bmatrix}
# $$
# such that $c^2 + s^2 = 1$. 
# More generally, we can use rotations on higher dimensional vectors by acting on two entries at a time.
# There are multiple ways of storing a rotation matrix, here we explore the simplest of storing just an angle.

# We will use a syntax in a struct that forces a field to be a special type. In what follows we define
# the `getindex` by first implementing multiplication, a pattern that will be reused in the questions



struct Rotation <: AbstractMatrix{Float64}
    θ::Float64 # The ::Float64 means θ can only be a Float64
end

import Base: *, size, getindex

size(Q::Rotation) = (2, 2)

function *(Q::Rotation, x::AbstractVector)
    if length(x) ≠ 2
        error("dimension mismatch")
    end
    θ = Q.θ
    c,s = cos(θ), sin(θ)
    a,b = x
    [c*a - s*b, s*a + c*b]
end

function getindex(Q::Rotation, k::Int, j::Int)
    e_k = zeros(2)
    e_j = zeros(2)
    e_k[k] = 1  # will error if k ≠ 0,1
    e_j[j] = 1  # will error if j ≠ 0,1
    e_k'*(Q*e_j)
end

Q = Rotation(0.1)

# We can test the ability to rotate a vector to the $x$-axis. Here we use the `atan(y,x)` function
# to compute the angle of a vector:


x = [-1,-2]
Q = Rotation(-atan(x[2], x[1]))
Q * x # first entry is norm(x), second entry is 0


# -----

# **Problem 1** Complete the implementation of `Rotations`, which represents an orthogonal matrix `Q` that is a product
# of rotations of angle `θ[k]`, each acting on the entries `k:k+1`. That is, it returns $Q = Q_1⋯Q_k$ such that
# $$
# Q_k[k:k+1,k:k+1] = 
# \begin{bmatrix}
# \cos θ[k] & -\sin θ[k]\\
# \sin θ[k] & \cos θ[k]
# \end{bmatrix}
# $$

struct Rotations <: AbstractMatrix{Float64}
    θ::Vector{Float64} # a vector of angles
end



## we use the number of rotations to deduce the dimensions of the matrix
size(Q::Rotations) = (length(Q.θ)+1, length(Q.θ)+1)

function *(Q::Rotations, x::AbstractVector)
    y = copy(x) # copies x to a new Vector 
    ## TODO: Apply Q in O(n) operations, modifying y in-place

    

    y
end

function getindex(Q::Rotations, k::Int, j::Int)
    ## TODO: Return Q[k,j] in O(n) operations using *.

    
end

θ = randn(5)
Q = Rotations(θ)
@test Q'Q ≈ I
@test Rotations([π/2, -π/2]) ≈ [0 0 -1; 1 0 0; 0 -1 0]


# ------

# ### III.5.2 Reflections


# We can also construct reflections. We do so first using the non-optimal approach of
# create a dense matrix. 



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


x = randn(3) + im*randn(3)
Q = dense_householderreflection(x)
Q * x # all the entries apart from the first are numerically zero

# A matrix-vector product is $O(n^2)$ operations but we know we can reduce it to $O(n)$.
# Thus we will create a special type to represent the reflection and obtain the better complexity
# multiplication. Because we want the matrix to be real when the entries are real we will use
# a special feature called "templating". Here by adding the `{T}` after the type we allow this to
# be either a `Float64` or `ComplexF64` (or indeed a `BigFloat`). We also do some checking
# to make sure that our defining vector is already normalised. 

struct Reflection{T} <: AbstractMatrix{T}
    v::Vector{T} # T can be either a Float64 or ComplexF64
end

function Reflection(v::Vector)
    T = eltype(v) # find the type of the entries of v
    if !(norm(v) ≈ 1)
        error("input must be normalised")
    end
    Reflection{T}(v) # create an instance of Reflection, specifying the entry type
end

# -----

# **Problem 2(a)** Complete the implementation of a type representing an n × n
# reflection that supports `Q[k,j]` in $O(1)$ operations and `*` in $O(n)$ operations.
# The reflection may be complex (that is, $Q ∈ U(n)$ is unitary).

## Represents I - 2v*v'


size(Q::Reflection) = (length(Q.v),length(Q.v))

## getindex(Q, k, j) is synonym for Q[k,j]
function getindex(Q::Reflection, k::Int, j::Int)
    ## TODO: implement Q[k,j] == (I - 2v*v')[k,j] but using O(1) operations.
    ## Hint: the function `conj` gives the complex-conjugate
    
end
function *(Q::Reflection, x::AbstractVector)
    ## TODO: implement Q*x, equivalent to (I - 2v*v')*x but using only O(n) operations
    
end

## If your code is correct, these "unit tests" will succeed
n = 10
x = randn(n) + im*randn(n)
v = x/norm(x)
Q = Reflection(v)
@test Q == I-2v*v'
@test Q'Q ≈ I


## We can scale to very large sizes. here we check the reflection property on an 100_000 matrix:
n = 100_000
x = randn(n) + im*randn(n)
v = x/norm(x)
Q = Reflection(v)
@test Q*x ≈ -x




# **Problem 2(b)** Complete the following implementation of a Housholder reflection  so that the
# unit tests pass, using the `Reflection` type created above.
# Here `s == true` means the Householder reflection is sent to the positive axis and `s == false` is the negative axis.

function householderreflection(s::Bool, x::AbstractVector)
    ## TODO: return a Reflection corresponding to a Householder reflection
    
end

x = randn(5)
Q = householderreflection(true, x)
@test Q isa Reflection
@test Q*x ≈ [norm(x);zeros(eltype(x),length(x)-1)]

Q = householderreflection(false, x)
@test Q isa Reflection
@test Q*x ≈ [-norm(x);zeros(eltype(x),length(x)-1)]



# **Problem 2(c)**
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


struct Reflections{T} <: AbstractMatrix{T}
    V::Matrix{T} # Columns of V are the householder vectors
end

size(Q::Reflections) = (size(Q.V,1), size(Q.V,1))


function *(Q::Reflections, x::AbstractVector)
    ## TODO: Apply Q in O(mn) operations by applying
    ## the reflection corresponding to each column of Q.V to x
    
    

    x
end

function getindex(Q::Reflections, k::Int, j::Int)
    ## TODO: Return Q[k,j] in O(mn) operations (hint: use *)

    
end

Y = randn(5,3)
V = Y * Diagonal([1/norm(Y[:,j]) for j=1:3])
Q = Reflections(V)
@test Q ≈ (I - 2V[:,1]*V[:,1]')*(I - 2V[:,2]*V[:,2]')*(I - 2V[:,3]*V[:,3]')
@test Q'Q ≈ I

# -----


# III.6 QR Factorisation

# III.6.2 Householder reflections and QR

# The proof by induction in the notes leads naturally to an iterative algorithm. Note that $\tilde Q$ is a product of all
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
# reflection this is a slow algorithm: it uses $O(m^2 n^2)$ operations, which is too many!
# By being more careful about how we apply and store reflections we can avoid this,
# in particular, taking advantage of the types `Reflection` and `Reflections`.

# ------

# **Problem 3** Complete the following function that implements
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
        
    end
    Q,R
end

A = randn(600,400)
Q,R = householderqr(A)
@test Q*R ≈ A


# ------ 

# An alternative to using reflections to introduce zeros is to use rotations.
# This is particularly convenient for tridiagonal matrices, where one needs to only
# make one sub-diagonal zero. Here we explore a tridiagonal QR built from rotations
# in a way that the factorisation can be computed in $O(n)$ operations.


# -----


# **Problem 4** This problem explores computing  a QR factorisation of a Tridiagonal matrix in $O(n)$ operations.
# This will introduce entries in the second super-diagonal, hence we will use the `UpperTridiagonal` type
# from Lab 6 (solution copied below). Complete the implementation of `bandedqr`, that only takes $O(n)$ operations,
# using an instance of `Reflections` to represent `Q` and `UpperTriangular` to represent `R`.

import Base: *, size, getindex, setindex!
struct UpperTridiagonal{T} <: AbstractMatrix{T}
    d::Vector{T}   # diagonal entries
    du::Vector{T}  # super-diagonal enries
    du2::Vector{T} # second-super-diagonal entries
end

size(U::UpperTridiagonal) = (length(U.d),length(U.d))

function getindex(U::UpperTridiagonal, k::Int, j::Int)
    d,du,du2 = U.d,U.du,U.du2
    if j - k == 0
        d[j]
    elseif j - k == 1
        du[k]
    elseif j - k == 2
        du2[k]
    else
        0
    end
end

function setindex!(U::UpperTridiagonal, v, k::Int, j::Int)
    d,du,du2 = U.d,U.du,U.du2
    if j > k+2
        error("Cannot modify off-band")
    end
    if j - k == 0
        d[k] = v
    elseif j - k == 1
        du[k] = v
    elseif j - k == 2
        du2[k] = v
    else
        error("Cannot modify off-band")
    end
    U # by convention we return the matrix
end


function bandedqr(A::Tridiagonal)
    n = size(A, 1)
    Q = Rotations(zeros(n - 1)) # Assume Float64
    R = UpperTridiagonal(zeros(n), zeros(n - 1), zeros(n - 2))

    ## TODO: Populate Q and R by looping through the columns of A.

    
    Q, R
end

A = Tridiagonal([1, 2, 3, 4], [1, 2, 3, 4, 5], [1, 2, 3, 4])
Q, R = bandedqr(A)
@test Q*R ≈ A



# III.6.3 QR and least squares

# When we type `A \ b` with a rectangular matrix `A` it is
# solving a least squares system. We can use the `qr` function 

A = randn(600,400)
b = randn(600)

Q,R̂ = qr(A)

# Here `Q` is a special type representing an orthogonal matrix.
# `R̂` is an `UpperTriangular`, that is, we only store the upper triangular
# entries of `R` (which is the same as the reduced QR factorisation). 
# Thus to solve a least squares problem we need to drop the extra entries as
# follows:

c = Q'b # invert Q
c̃ = c[1:size(R̂,1)] # drop extra entries
A \ b ≈ R\c̃

# **Problem 5** Complete the function `leastsquares(A, b)` that uses your
# `householderqr` function to solve a least squares problem.

function leastsquares(A, b)
    ## TODO: use householderqr to solve a least squares problem.
    
end