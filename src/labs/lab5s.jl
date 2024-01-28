# # MATH50003 (2022–23)
# # Lab 5: III.1 Structured Matrices and III.2 Differential Equations


# In this lab we explore the construction of vectors and matrices, in particular those with sparsity structure
# which we capture using special types. We also explore the reduction of differential equations to
# banded linear systems. 


## LinearAlgebra contains routines for doing linear algebra
## BenchmarkTools is a package for reliable timing
using LinearAlgebra, Plots, BenchmarkTools, Test


# **Remark** One should normally not need to implement these methods oneself as there
# are packages available, e.g. [DifferentialEquations.jl](https://github.com/SciML/DifferentialEquations.jl). Moreover Forward and Backward
# Euler are only the first baby steps to a wide range of time-steppers, with Runge–Kutta being
# one of the most successful.
# For example we can solve
# a simple differential equation like a pendulum $u'' = -\sin u$ can be solved
# as follows (writing at a system $u' = v, v' = -\sin u$):

using DifferentialEquations, LinearAlgebra, Plots

u = solve(ODEProblem((u,_,x) -> [u[2], -sin(u[1])], [1,0], (0,10)))
plot(u)

# However, even in these automated packages one has a choice of different methods with
# different behaviour, so it is important to understand on a mathematical level what is happening under the hood.


# **Learning Outcomes**
#
# Mathematical knowledge:
#
# 1. Banded matrices and their utilisation them for better complexity linear algebra.
# 2. Reduction of differential equations to linear systems
# 3. Two-point boundary value problems
#
# Coding knowledge:
#
# 1. Construction of a dense `Vector` or `Matrix` either directly or via comprehensions or broadcasting
# 2. The `vec`, `transpose`, `zeros`, `ones` and `fill` functions.
# 3. Using `\` to solve linear systems.


# ## III.1 Structured Matrices

# Before discussing structured matrices we give an overview of creating arrays  (vectors and matrices)
# in Julia.

# ### III.1.1 Dense matrices


# One can create arrays in multiple ways. For example, the function `zeros(Int, 10)` creates
# a 10-element `Vector` whose entries are all `zero(Int) == 0`. Or `fill(x, 10)` creates a 
# 10-element `Vector` whose entries are all equal to `x`. Or you can use a comprehension:
# for example `[k^2 for k = 1:10]` creates a vector whose entries are `[1^2, 2^2, …, 10^2]`.
# This also works for matrices: `zeros(Int, 10, 5)` creates a 10 × 5 matrix of all zeros,
# and `[k^2 + j for k=1:3, j=1:4]` creates the following:

[k^2 + j for k=1:3, j=1:4] # k is the row, j is the column

# Note sometimes it is best to create a vector/matrix and populate it. For example, the
# previous matrix could also been constructed as follows:

A = zeros(Int, 3, 4) # create a 3 × 4 matrix whose entries are `Int`
for k = 1:3, j = 1:4
    A[k,j] = k^2 + j # set the entries of A
end
A

# Be careful: a `Matrix` or `Vector` can only ever contain entries of the right
# type. It will attempt to convert an assignment to the right type but will throw
# an error if not successful:

A[2,3] = 2.0 # works because 2.0 is a Float64 that is exactly equal to an Int
A[1,2] = 2.3 # fails since 2.3 is a Float64 that cannot be converted to an Int


# **Problem 1(a)** Create a 5×6 matrix whose entries are `Int` which is
# one in all entries. Hint: use a for-loop, `ones`, `fill`, or a comprehension.
## TODO: 
## SOLUTION

## 1. For-loop:

ret = zeros(Int, 5, 6)
for k=1:5, j=1:6
    ret[k,j] = 1
end
ret

## 2. Ones:

ones(Int, 5, 6)

## 3. Fill:

fill(1, 5, 6)

## 4. Comprehension:

[1 for k=1:5, j=1:6]


## END

# **Problem 1(b)** Create a 1 × 5 `Matrix{Int}` with entries `A[k,j] = j`. Hint: use a for-loop or a comprehension.

## TODO: 
## SOLUTION

## 1. For-loop

A = zeros(Int, 1, 5)
for j = 1:5
    A[1,j] = j
end

## 2. Comprehension

[j for k=1:1, j=1:5]

## 3. convert transpose:

## Note: (1:5)' is a "row-vector" which behaves differently than a matrix
Matrix((1:5)')


## END

# **Problem 1(c)** Create a vector of length 5 whose entries are `Float64`
# approximations of `exp(-k)`. Hint: one use a for-loop or broadcasting `f.(x)` notation.
## TODO: 
## SOLUTION

## 1. For-loop
v = zeros(5) # defaults to Float64
for k = 1:5
    v[k] = exp(-k)
end

## 2. Broadcast:
exp.(-(1:5))

## 3. Explicit broadcsat:
broadcast(k -> exp(-k), 1:5)

## 4. Comprehension:
[exp(-k) for k=1:5]


## END

# **Problem 1(d)** Create a 5 × 6 matrix `A` whose entries `A[k,j] == cos(k+j)`.
## TODO: 

## SOLUTION

#  1. For-loop:

A = zeros(5,6)
for k = 1:5, j = 1:6
    A[k,j] = cos(k+j)
end
A

# 2. Broadcasting:

k = 1:5
j = 1:6
cos.(k .+ j')

# 3. Broadcasting (explicit):

broadcast((k,j) -> cos(k+j), 1:5, (1:6)')

## END


# A  `Matrix` is stored consecutively in memory, going down column-by-
# column (_column-major_). That is,

A = [1 2;
     3 4;
     5 6]

# Is actually stored equivalently to a length `6` vector:

vec(A)

# which in this case would be stored using `8 * 6 = 48` consecutive bytes.
# Behind the scenes, a matrix is a "pointer" to the location of the first entry alongside two integers
# dictating the row and column sizes.


# Matrix-vector multiplication works as expected:

x = [7, 8]
A * x

# Note there are two ways this can be implemented: 


# In code this can be implemented for any types that support `*` and `+` as follows:

function mul_rows(A, x)
    m,n = size(A)
    # promote_type type finds a type that is compatible with both types, eltype gives the type of the elements of a vector / matrix
    T = promote_type(eltype(x), eltype(A))
    c = zeros(T, m) # the returned vector, begins of all zeros
    for k = 1:m, j = 1:n
        c[k] += A[k, j] * x[j] # equivalent to c[k] = c[k] + A[k, j] * x[j]
    end
    c
end



function mul_cols(A, x)
    m,n = size(A)
    # promote_type type finds a type that is compatible with both types, eltype gives the type of the elements of a vector / matrix
    T = promote_type(eltype(x), eltype(A))
    c = zeros(T, m) # the returned vector, begins of all zeros
    for j = 1:n, k = 1:m
        c[k] += A[k, j] * x[j] # equivalent to c[k] = c[k] + A[k, j] * x[j]
    end
    c
end


# Both implementations match exactly for integer inputs:

mul_rows(A, x), mul_cols(A, x) # also matches `A*x`


# Either implementation will be $O(mn)$ operations. However, the implementation 
# `mul_cols` accesses the entries of `A` going down the column,
# which happens to be _significantly faster_ than `mul_rows`, due to accessing
# memory of `A` in order. We can see this by measuring the time it takes using `@btime`:

n = 1000
A = randn(n,n) # create n x n matrix with random normal entries
x = randn(n) # create length n vector with random normal entries

@btime mul_rows(A,x)
@btime mul_cols(A,x)
@btime A*x; # built-in, high performance implementation. USE THIS in practice

# Here `ms` means milliseconds (`0.001 = 10^(-3)` seconds) and `μs` means microseconds (`0.000001 = 10^(-6)` seconds).
# So we observe that `mul` is roughly 3x faster than `mul_rows`, while the optimised `*` is roughly 5x faster than `mul`.



# Note that the rules of floating point arithmetic apply here: matrix multiplication with floats
# will incur round-off error (the precise details of which are subject to the implementation):


A = [1.4 0.4;
     2.0 1/2]
A * [1, -1] # First entry has round-off error, but 2nd entry is exact

# And integer arithmetic will be subject to overflow:

A = fill(Int8(2^6), 2, 2) # make a matrix whose entries are all equal to 2^6
A * Int8[1,1] # we have overflowed and get a negative number -2^7

# Solving a linear system is done using `\`:

A = [1 2 3;
     1 2 4;
     3 7 8]
b = [10; 11; 12]
A \ b

# Despite the answer being integer-valued, 
# here we see that it resorted to using floating point arithmetic,
# incurring rounding error. 
# But it is "accurate to (roughly) 16-digits".
# As we shall see, the way solving a linear system works is we first write `A` as a
# product of matrices that are easy to invert, e.g., a product of triangular matrices or a product of an orthogonal
# and triangular matrix.


# The following problem compares the behaviour of `mul_cols` defined in lectures

function mul_cols(A, x)
    m,n = size(A)
    c = zeros(eltype(x), m) # eltype is the type of the elements of a vector/matrix
    for j = 1:n, k = 1:m
        c[k] += A[k, j] * x[j]
    end
    c
end

# to the inbuilt matrix-vector multiplication operation `A*x`. The point is that
# sometimes the choice of algorithm, despite being mathematically equivalent, can change the exact results
# when using floating point.

# **Problem 2** Show that `A*x` is not
# implemented as `mul_cols(A, x)` from the lecture notes
# by finding a `Float64` example  where the bits do not match.
# Hint: either guess-and-check, perhaps using `randn(n,n)` to make a random `n × n` matrix.


## SOLUTION

## Then we can easily find examples, in fact we can write a function that searches for examples:

using ColorBitstring

function findblasmuldifference(n, l)
	for j = 1:n
		A = randn(l,l)
		x = rand(l)
		if A*x != mul_cols(A,x) 
			return (A,x)
		end
	end
end

n = 100 # number of attempts
l = 10 # size of objects
A,x = findblasmuldifference(n,l) # find a difference

println("Bits of obtained A*x")
printlnbits.(A*x);
println("Bits of obtained mul_cols(A,x)")
printlnbits.(mul_cols(A,x));
println("Difference vector between the two solutions:")
println(A*x-mul_cols(A,x))

## END

# We can also transpose a matrix `A`, This is done lazily 
# and so `transpose(A)` (which is equivalent to the adjoint/conjugate-transpose
# `A'` when the entries are real),
# is just a special type with a single field: `transpose(A).parent == A`.
# This is equivalent to 
# _row-major_ format, where the next address in memory of `transpose(A)` corresponds to
# moving along the row.

# ## 3. Triangular Matrices

# In lectures we covered algorithms involving upper-triangular matrices. Here we want to implement
# the lower-triangular analogues.

# **Problem 3.1** Complete the following function for lower triangular matrix-vector
# multiplication without ever accessing the zero entries of `L` above the diagonal.
# Hint: just copy code for `mul_cols` and modify the for-loop ranges a la the `UpperTriangular`
# case.

function mul_cols(L::LowerTriangular, x)
    n = size(L,1)

    ## promote_type type finds a type that is compatible with both types, eltype gives the type of the elements of a vector / matrix
    T = promote_type(eltype(x),eltype(L))
    b = zeros(T,n) # the returned vector, begins of all zeros

    ## TODO: populate b so that L*x ≈ b
    ## SOLUTION
    for j = 1:n, k = j:n
        b[k] += L[k, j] * x[j]
    end
    ## END

    b
end

L = LowerTriangular(randn(5,5))
x = randn(5)
@test L*x ≈ mul_cols(L, x)


# **Problem 3.2** Complete the following function for solving linear systems with
# lower triangular systems by implementing forward-substitution.


function ldiv(L::LowerTriangular, b)
    n = size(L,1)
    
    if length(b) != n
        error("The system is not compatible")
    end
        
    x = zeros(n)  # the solution vector
    ## TODO: populate x using forward-substitution so that L*x ≈ b
    ## SOLUTION
    for k = 1:n  # start with k = 1
        r = b[k]  # dummy variable
        for j = 1:k-1
            r -= L[k,j]*x[j]
        end
        x[k] = r/L[k,k]
    end
    ## END
    x
end


L = LowerTriangular(randn(5,5))
b = randn(5)
@test L\b ≈ ldiv(L, b)


# ## 4. Banded matrices

# Banded matrices are very important in differential equations and enable much faster algorithms. 
# Here we look at banded upper triangular matrices by implementing a type that encodes this
# property:

struct UpperTridiagonal{T} <: AbstractMatrix{T}
    d::Vector{T}   # diagonal entries: d[k] == U[k,k]
    du::Vector{T}  # super-diagonal enries: du[k] == U[k,k+1]
    du2::Vector{T} # second-super-diagonal entries: du2[k] == U[k,k+2]
end

# This uses the notation `<: AbstractMatrix{T}`: this tells Julia that our type is in fact a matrix.
# In order for it to behave a matrix we have to overload the function `size` for our type to return
# the dimensions (in this case we just use the length of the diagonal):

size(U::UpperTridiagonal) = (length(U.d),length(U.d))

# Julia still doesn't know what the entries of the matrix are. To do this we need to overload `getindex`.
# We also overload `setindex!` to allow changing the non-zero entries.

# **Problem 4.1** Complete the implementation of `UpperTridiagonal` which represents a banded matrix with
# bandwidths $(l,u) = (0,2)$ by overloading `getindex` and `setindex!`. Return zero (of the same type as the other entries)
# if we are off the bands.

## getindex(U, k, j) is another way to write U[k,j].
## This function will therefore be called when we call U[k,j]
function getindex(U::UpperTridiagonal, k::Int, j::Int)
    d,du,du2 = U.d,U.du,U.du2
    ## TODO: return U[k,j]
    ## SOLUTION
    if j == k+2
    	return U.du2[k]    
    elseif j == k+1
    	return U.du[k]
    elseif j == k
    	return U.d[k]
    else # off band entries are zero
    	return zero(eltype(U))
    end
    ## END
end

## setindex!(U, v, k, j) gets called when we write (U[k,j] = v).
function setindex!(U::UpperTridiagonal, v, k::Int, j::Int)
    d,du,du2 = U.d,U.du,U.du2
    if j > k+2 || j < k
        error("Cannot modify off-band")
    end

    ## TODO: modify d,du,du2 so that U[k,j] == v
    ## SOLUTION
    if j == k+2
    	du2[k] = v  
    elseif j == k+1
    	du[k] = v
    elseif j == k
    	d[k] = v
    end
    ## END
    U # by convention we return the matrix
end

U = UpperTridiagonal([1,2,3,4,5], [1,2,3,4], [1,2,3])
@test U == [1 1 1 0 0;
            0 2 2 2 0;
            0 0 3 3 3;
            0 0 0 4 4;
            0 0 0 0 5]

U[3,4] = 2
@test U == [1 1 1 0 0;
            0 2 2 2 0;
            0 0 3 2 3;
            0 0 0 4 4;
            0 0 0 0 5]




# **Problem 4.2** Complete the following implementations of `*` and `\` for `UpperTridiagonal` so that
# they take only $O(n)$ operations. Hint: the function `max(a,b)` returns the larger of `a` or `b`
# and `min(a,b)` returns the smaller. They may help to avoid accessing zeros.

function *(U::UpperTridiagonal, x::AbstractVector)
    n = size(U,1)
    ## promote_type type finds a type that is compatible with both types, eltype gives the type of the elements of a vector / matrix
    T = promote_type(eltype(x),eltype(U))
    b = zeros(T, n) # the returned vector, begins of all zeros
    ## TODO: populate b so that U*x ≈ b (up to rounding)
    ## SOLUTION
    for j = 1:n, k = max(j-2,1):j
        b[k] += U[k, j] * x[j]
    end
    ## END
    b
end

function \(U::UpperTridiagonal, b::AbstractVector)
    n = size(U,1)
    T = promote_type(eltype(b),eltype(U))

    if length(b) != n
        error("The system is not compatible")
    end
        
    x = zeros(T, n)  # the solution vector
    ## TODO: populate x so that U*x ≈ b
    ## SOLUTION
    for k = n:-1:1  # start with k=n, then k=n-1, ...
        r = b[k]  # dummy variable
        for j = k+1:min(n, k+2)
            r -= U[k,j]*x[j] # equivalent to r = r - U[k,j]*x[j]
        end
        ## after this for loop, r = b[k] - ∑_{j=k+1}^n U[k,j]x[j]  
        x[k] = r/U[k,k]
    end
    ## END
    x
end

n = 1_000_000 # under-scores are like commas: so this is a million: 1,000,000
U = UpperTridiagonal(ones(n), fill(0.5,n-1), fill(0.1,n-2))
x = ones(n)
b = [fill(1.6,n-2); 1.5; 1] # exact result
## note following should take much less than a second
@test U*x ≈ b
@test U\b ≈ x



#-----





# ## 2. Triangular matrices

# Triangular matrices are represented by dense square matrices where the entries below the
# diagonal are ignored:

A = [1 2 3;
     4 5 6;
     7 8 9]
U = UpperTriangular(A)

# We can see that `U` is storing all the entries of `A` in a field called `data`:

U.data

# Similarly we can create a lower triangular matrix by ignoring the entries above the diagonal:

L = LowerTriangular(A)

# If we know a matrix is triangular we can do matrix-vector multiplication in roughly half
# the number of operations by skipping over the entries we know are zero:

# **Algorithm 3 (upper-triangular matrix-vector multiplication by columns)**

function mul_cols(U::UpperTriangular, x)
    n = size(U,1)
    # promote_type type finds a type that is compatible with both types, eltype gives the type of the elements of a vector / matrix
    T = promote_type(eltype(x),eltype(U))
    b = zeros(T, n) # the returned vector, begins of all zeros
    for j = 1:n, k = 1:j # k = 1:j instead of 1:m since we know U[k,j] = 0 if k > j.
        b[k] += U[k, j] * x[j]
    end
    b
end

x = [10, 11, 12]
# matches built-in *
@test mul_cols(U, x) == U*x



# Moreover, we can easily invert matrices. 
# Consider a simple 3×3 example, which can be solved with `\`:

b = [5, 6, 7]
x = U \ b # Excercise: why does this return a float vector?

# Behind the seens, `\` is doing back-substitution: considering the last row, we have all
# zeros apart from the last column so we know that `x[3]` must be equal to:

b[3] / U[3,3]

# Once we know `x[3]`, the second row states `U[2,2]*x[2] + U[2,3]*x[3] == b[2]`, rearranging
# we get that `x[2]` must be:

(b[2] - U[2,3]*x[3])/U[2,2]

# Finally, the first row states `U[1,1]*x[1] + U[1,2]*x[2] + U[1,3]*x[3] == b[1]` i.e.
# `x[1]` is equal to

(b[1] - U[1,2]*x[2] - U[1,3]*x[3])/U[1,1]


# More generally, we can solve the upper-triangular system using _back-substitution_:


# In code this can be implemented for any types that support `*`, `+` and `/` as follows:

# ldiv(U, b) is our implementation of U\b
function ldiv(U::UpperTriangular, b)
    n = size(U,1)
    
    if length(b) != n
        error("The system is not compatible")
    end
        
    x = zeros(n)  # the solution vector
    
    for k = n:-1:1  # start with k=n, then k=n-1, ...
        r = b[k]  # dummy variable
        for j = k+1:n
            r -= U[k,j]*x[j] # equivalent to r = r - U[k,j]*x[j]
        end
        # after this for loop, r = b[k] - ∑_{j=k+1}^n U[k,j]x[j]  
        x[k] = r/U[k,k]
    end
    x
end

@test ldiv(U, x) ≈ U\x


# Diagonal matrices in Julia are stored as a vector containing the diagonal entries:

x = [1,2,3]
D = Diagonal(x) # the type Diagonal has a single field: D.diag

# It is clear that we can perform diagonal-vector multiplications and solve linear systems involving diagonal matrices efficiently
# (in $O(n)$ operations).


# We can create Bidiagonal matrices in Julia by specifying the diagonal and off-diagonal:


L = Bidiagonal([1,2,3], [4,5], :L) # the type Bidiagonal has three fields: L.dv (diagonal), L.ev (lower-diagonal), L.uplo (either 'L', 'U')
##
Bidiagonal([1,2,3], [4,5], :U)


# Multiplication and solving linear systems with Bidiagonal systems is also $O(n)$ operations, using the standard
# multiplications/back-substitution algorithms but being careful in the loops to only access the non-zero entries. 


# Julia has a type `Tridiagonal` for representing a tridiagonal matrix from its sub-diagonal, diagonal, and super-diagonal:

T = Tridiagonal([1,2], [3,4,5], [6,7]) # The type Tridiagonal has three fields: T.dl (sub), T.d (diag), T.du (super)

# Tridiagonal matrices will come up in solving second-order differential equations and orthogonal polynomials.
# We will later see how linear systems involving tridiagonal matrices can be solved in $O(n)$ operations.


# **Example**

# Let's do an example of integrating $\cos x$, and see if our method matches
# the true answer of $\sin x$. First we construct the system
# as a lower-triangular, `Bidiagonal` matrix:


function indefint(x)
    h = step(x) # x[k+1]-x[k]
    n = length(x)
    L = Bidiagonal([1; fill(1/h, n-1)], fill(-1/h, n-1), :L)
end

n = 10
x = range(0, 1; length=n)
L = indefint(x)

# We can now solve for our particular problem using both the left and 
# mid-point rules:

c = 0 # u(0) = 0
f = x -> cos(x)


m = (x[1:end-1] + x[2:end])/2 # midpoints


𝐟ᶠ = f.(x[1:end-1]) # evaluate f at all but last points
𝐟ᵐ = f.(m)          # evaluate f at mid-points
𝐮ᶠ = L \ [c; 𝐟ᶠ] # integrate using forward-differences
𝐮ᵐ = L \ [c; 𝐟ᵐ] # integrate using central-differences

plot(x, sin.(x); label="sin(x)", legend=:bottomright)
scatter!(x, 𝐮ᶠ; label="forward")
scatter!(x, 𝐮ᵐ; label="mid")

# They both are close though the mid-point version is significantly
# more accurate.
#  We can estimate how fast it converges:

# Error from indefinite integration with c and f
function forward_err(u, c, f, n)
    x = range(0, 1; length = n)
    uᶠ = indefint(x) \ [c; f.(x[1:end-1])]
    norm(uᶠ - u.(x), Inf)
end

function mid_err(u, c, f, n)
    x = range(0, 1; length = n)
    m = (x[1:end-1] + x[2:end]) / 2 # midpoints
    uᵐ = indefint(x) \ [c; f.(m)]
    norm(uᵐ - u.(x), Inf)
end

ns = 10 .^ (1:8) # solve up to n = 10 million
scatter(ns, forward_err.(sin, 0, f, ns); xscale=:log10, yscale=:log10, label="forward")
scatter!(ns, mid_err.(sin, 0, f, ns); label="mid")
plot!(ns, ns .^ (-1); label="1/n")
plot!(ns, ns .^ (-2); label="1/n^2")
```
This is a log-log plot:we scale both $x$ and $y$ axes logarithmically so that
$n^α$ becomes a straight line where the slope is dictated by $α$.
We seem experimentally that the error for forward-difference is $O(n^{-1})$
while for mid-point/central-differences we get faster $O(n^{-2})$ convergence. 
Both methods appear to be stable.
 

Here is a simple example for solving:
    $$
    u'(0) = 1, u' + t u = {\rm e}^t
    $$
    which has an exact solution in terms of a special error function
    (which we determined using Mathematica).
    
    ```julia
    using SpecialFunctions
    c = 1
    a = t -> t
    n = 2000
    t = range(0, 1; length=n)
    # exact solution, found in Mathematica
    u = t -> -(1/2)*exp(-(1+t^2)/2)*(-2sqrt(ℯ) + sqrt(2π)erfi(1/sqrt(2)) - sqrt(2π)erfi((1 + t)/sqrt(2)))
    
    h = step(t)
    L = Bidiagonal([1; fill(1/h, n-1)], a.(t[1:end-1]) .- 1/h, :L)
    
    norm(L \ [c; exp.(t[1:end-1])] - u.(t),Inf)
    ```
    We see that it is converging to the true result.
    
    
    Note that this is a simple forward-substitution of a bidiagonal system,
    so we can also just construct it directly:
    $$
    \begin{align*}
    u_1 &= c \\
    u_{k+1} &= (1 + h a(t_k)) u_k + h f(t_k)
    \end{align*}
    $$
    
    
    **Remark (advanced)** Note this can alternatively be reduced to an integral
    $$
    u(t) = c \hbox{e}^{a t} + \hbox{e}^{a t} \int_0^t f(τ) \hbox{e}^{-a τ} \hbox d τ
    $$
    and solved as above but this approach is harder to generalise.


    ## Systems of equations

We can also solve systems, that is, equations of the form:
$$
\begin{align*}
𝐮(0) &= 𝐜 \\
𝐮'(t) - A(t) 𝐮(t) &= 𝐟(t)
\end{align*}
$$
where $𝐮, 𝐟 : [0,T] → ℝ^d$ and $A : [0,T] → ℝ^{d × d}$.
We again discretise at the grid $t_k$
by approximating $𝐮(t_k) ≈ 𝐮_k ∈ ℝ^d$.
This can be reduced to a block-bidiagonal system as in
the scalar case which is solved via forward-substitution. Though
it's easier to think of it directly. 

Forward Euler gives us:
$$
\begin{align*}
𝐮_1 &= c \\
𝐮_{k+1} &= 𝐮_k + h A(t_k) 𝐮_k + h 𝐟(t_k)
\end{align*}
$$
That is, each _time-step_ consists of matrix-vector multiplication.
On the other hand Backward Euler requires inverting a matrix
at each time-step:
$$
\begin{align*}
𝐮_1 &= c \\
𝐮_{k+1} &= (I- h A(t_{k+1}))^{-1} (𝐮_k  + h 𝐟(t_{k+1}))
\end{align*}
$$


**Example (Airy equation)**
Consider the (negative-time) Airy equation:
$$
\begin{align*}
u(0) &= 1 \\
u'(0) &= 0 \\
u''(t) + t u &= 0
\end{align*}
$$
We can recast it as a system by defining
$$
𝐮(x) = \begin{bmatrix} u(x) \\ u'(x) \end{bmatrix}
$$
which satisfies
$$
\begin{align*}
𝐮(0) = \begin{bmatrix} 1 \\ 0 \end{bmatrix} \\
𝐮' - \begin{bmatrix} 0 & 1 \\ -t & 0 \end{bmatrix} 𝐮 = 𝟎.
\end{align*}
$$
It is natural to represent the _time-slices_ $𝐮_k$ as
columns of a matrix $U = [𝐮_1 | ⋯ | 𝐮_n] ∈ ℝ^{2 × n}$. Thus we get:
```julia
n = 100_000
t = range(0, 50; length=n)
A = t -> [0 1; -t 0]
h = step(t)

U = zeros(2, n) # each column is a time-slice
U[:,1] = [1.0,0.0] # initial condition
for k = 1:n-1
    U[:,k+1] = (I + h*A(t[k]))*U[:,k]
end

# We plot both the value and its derivative
plot(t, U')
```

We leave implementation of backward Euler as a simple exercise.

**Example (Heat on a graph)**
Those who took Introduction to Applied Mathematics will recall
heat equation on a graph. Consider a simple graph of $m$ nodes labelled
$1, …, m$ where node $k$ is connected to neighbouring nodes ${k-1}$ and ${k+1}$,
whereas node $1$ is only connected to node $2$ and node $m$ only connected to
${m-1}$. The graph Laplacian corresponding to this system is the matrix:
$$
Δ := \begin{bmatrix} -1 & 1 \\ 
            1 & -2 & ⋱ \\ 
            & 1 & ⋱ & 1 \\
            && ⋱ & -2 & 1 \\
                &&& 1 & -1
                \end{bmatrix}
$$
If we denote the heat at time $t$ at node $k$ as $u_k(t)$,
which we turn into a vector
$$
𝐮(t) = \begin{bmatrix} u_1(t) \\ ⋮ \\ u_m(t) \end{bmatrix}
$$
We consider the case of a periodic forcing at the middle node $n = ⌊m/2⌋$.

Heat equation on this lattice is defined as follows:
$$
𝐮' = Δ𝐮 + 𝐞_{⌊m/2⌋} \cos ωt
$$
We can employ forward and backward Euler:
```julia
n = 1_000 # number of time-steps
t = range(0, 100; length=n)
h = step(t)

m = 50 # number of nodes


Δ = SymTridiagonal([-1; fill(-2.0, m-2); -1], ones(m-1))
ω = 1
f = t -> cos(ω*t) # periodic forcing with period 1

Uᶠ = zeros(m, n) # each column is a time-slice for forward Euler
Uᵇ = zeros(m, n) # each column is a time-slice for backwar Euler

Uᶠ[:,1] = Uᵇ[:,1] = zeros(m) # initial condition



for k = 1:n-1
    Uᶠ[:,k+1] = (I + h*Δ)*Uᶠ[:,k]
    Uᶠ[m÷2,k+1] += h*f(t[k]) # add forcing at 𝐞_1
end

𝐞 = zeros(m); 𝐞[m÷2] = 1;

for k = 1:n-1
    Uᵇ[:,k+1] = (I - h*Δ)\(Uᵇ[:,k] + h*f(t[k+1])𝐞)
end

# Uᶠ[:,end] is the solution at the last time step
scatter(Uᶠ[:,end]; label="forward")
scatter!(Uᵇ[:,end]; label="backward")
```
Both match! 

**Remark** If you change the number of time-steps to be too small, for example `n = 100`, forward
Euler blows up while backward Euler does not. This will be discussed in the problem
sheet.

**Remark (advanced)** Memory allocations are very expensive so
in practice one should preallocate and use memory. 


## Nonlinear problems

Forward-Euler extends naturally to nonlinear equations, 
including the
vector case:
$$
𝐮' = f(t, 𝐮(t))
$$
becomes:
$$
𝐮_{k+1} = 𝐮_k + h f(t_k, 𝐮_k)
$$
Here we show a simple solution to a nonlinear Pendulum:
$$
u'' = -\sin u
$$
by writing $𝐮(t) := [u_1(t),u_2(t)] :=[u(t),u'(t)]$ we have:
$$
𝐮'(t) =  \underbrace{\begin{bmatrix} u_2(t) \\ -\sin u_1(t) \end{bmatrix}}_{f(t, 𝐮(t))}
$$
Again we put the time slices into a $2 × n$ matrix:

```julia
n = 10_000
Uᶠ = zeros(2, n)
t = range(0, 20; length=n)
h = step(t) # same as x[k+1]-x[k]

Uᶠ[:,1] = [1,0] # initial condition
for k = 1:n-1
    u₁, u₂ = Uᶠ[:,k]
    Uᶠ[:,k+1] = [u₁, u₂] + h * [u₂,-sin(u₁)]
end

# just plot solution
plot(t, Uᶠ[1,:]; label="Pendulum angle")
```
As we see it roughly predicts the oscillatory behaviour of
a pendulum, and matches the simulation using DifferentialEquations.jl
above. However, over time there is an increase as we have not
resolved the solution. Increasing `n` further will cause the error
to decrease as the method does indeed converge.


## Poisson equation

Thus we solve:
```julia
x = range(0, 1; length = n)
h = step(x)
T = Tridiagonal([fill(1/h^2, n-2); 0], [1; fill(-2/h^2, n-2); 1], [0; fill(1/h^2, n-2)])
u = T \ [1; exp.(x[2:end-1]); 2]
scatter(x, u)
```

We can test convergence on $u(x) = \cos x^2$ which satisfies
$$
\begin{align*}
u(0) = 1 \\
u(1) = \cos 1 \\
u''(x) = -4x^2*\cos(x^2) - 2\sin(x^2)
\end{align*}
$$
We observe uniform ($∞$-norm) convergence:
```julia
function poisson_err(u, c_0, c_1, f, n)
    x = range(0, 1; length = n)
    h = step(x)
    T = Tridiagonal([fill(1/h^2, n-2); 0], [1; fill(-2/h^2, n-2); 1], [0; fill(1/h^2, n-2)])
    uᶠ = T \ [c_0; f.(x[2:end-1]); c_1]
    norm(uᶠ - u.(x), Inf)
end

u = x -> cos(x^2)
f = x -> -4x^2*cos(x^2) - 2sin(x^2)

ns = 10 .^ (1:8) # solve up to n = 10 million
scatter(ns, poisson_err.(u, 1, cos(1), f, ns); xscale=:log10, yscale=:log10, label="error")
plot!(ns, ns .^ (-2); label="1/n^2")
```




## 2. Indefinite integration

**Problem 2.1 (B)** Implement backward differences to approximate
indefinite-integration. How does the error compare to forward
and mid-point versions  for $f(x) = \cos x$ on the interval $[0,1]$?
Use the method to approximate the integrals of
$$
\exp(\exp x \cos x + \sin x), \prod_{k=1}^{1000} \left({x \over k}-1\right), \hbox{ and } f^{\rm s}_{1000}(x)
$$
to 3 digits, where $f^{\rm s}_{1000}(x)$ was defined in PS2.

**SOLUTION**

We can implement the backward difference solution as follows:
```julia
c = 0 # u(0) = 0
f = x -> cos(x)
n = 10

x = range(0,1;length=n)
h=step(x)
A = Bidiagonal([1; fill(1/h, n-1)], fill(-1/h, n-1), :L)
ub = A\[c; f.(x[2:end])]

##adding the forward and midpoint solutions here as well for comparison
m = (x[1:end-1] + x[2:end])/2

uf = A \ [c; f.(x[1:end-1])]
um = A \ [c; f.(m)]

plot(x, sin.(x); label="sin(x)", legend=:bottomright)
scatter!(x, ub; label="backward")
scatter!(x, um; label="mid")
scatter!(x, uf; label="forward")
```

Comparing each method's errors, we see that the backward method has the same error as the forward method:

```julia
function indefint(x)
    h = step(x) # x[k+1]-x[k]
    n = length(x)
    L = Bidiagonal([1; fill(1/h, n-1)], fill(-1/h, n-1), :L)
end

function forward_err(u, c, f, n)
    x = range(0, 1; length = n)
    uᶠ = indefint(x) \ [c; f.(x[1:end-1])]
    norm(uᶠ - u.(x), Inf)
end

function mid_err(u, c, f, n)
    x = range(0, 1; length = n)
    m = (x[1:end-1] + x[2:end]) / 2 # midpoints
    uᵐ = indefint(x) \ [c; f.(m)]
    norm(uᵐ - u.(x), Inf)
end

function back_err(u, c, f, n)
    x = range(0,1;length=n)
    h=step(x)
    A = Bidiagonal([1; fill(1/h, n-1)], fill(-1/h, n-1), :L)
    ub = A\[c; f.(x[2:end])]
    norm(ub - u.(x), Inf)
end

c = 0 # u(0) = 0
f = x -> cos(x)
m = (x[1:end-1] + x[2:end])/2 # midpoints
ns = 10 .^ (1:8) # solve up to n = 10 million


scatter(ns, forward_err.(sin, 0, f, ns); xscale=:log10, yscale=:log10, label="forward")
scatter!(ns, mid_err.(sin, 0, f, ns); label="mid")
scatter!(ns, back_err.(sin, 0, f, ns); label="back",alpha=0.5)
plot!(ns, ns .^ (-1); label="1/n")
plot!(ns, ns .^ (-2); label="1/n^2")
```

Part two:

```julia
c = 0 # u(0) = 0
n = 10000

#functions defined in the solutions to problem sheet 2
f = x -> exp(exp(x)cos(x) + sin(x))
g = x -> prod([x] ./ (1:1000) .- 1)
function cont(n, x)
    ret = 2*one(x)
    for k = 1:n-1
        ret = 2 + (x-1)/ret
    end
    1 + (x-1)/ret
end

x = range(0,1;length=n)
h=step(x)
A = Bidiagonal([1; fill(1/h, n-1)], fill(-1/h, n-1), :L)
uf = A\[c; f.(x[2:end])]
ug = A\[c; g.(x[2:end])]
ucont = A\[c; cont.(1000, x[2:end])]

uf_int = uf[end]
ug_int = ug[end]
ucont_int = ucont[end]

println("first function: ")
println(uf_int)
println("second functions: ")
println(ug_int)
println("third function: ")
println(ucont_int)
```

**Problem 2.2 (A)** Implement indefinite-integration 
where we take the average of the two grid points:
$$
{u'(x_{k+1}) + u'(x_k) \over 2} ≈ {u_{k+1} - u_k \over h}
$$
What is the observed rate-of-convergence using the ∞-norm for $f(x) = \cos x$
on the interval $[0,1]$?
Does the method converge if the error is measured in the $1$-norm?

**SOLUTION**

The implementation is as follows:

```julia
n = 10
x = range(0, 1; length=n+1)
h = 1/n
A = Bidiagonal([1; fill(1/h, n)], fill(-1/h, n), :L)
c = 0 # u(0) = 0
f = x -> cos(x)

𝐟 = f.(x) # evaluate f at all but last points
uₙ = A \ [c; (𝐟[1:end-1] + 𝐟[2:end])/2]

plot(x, sin.(x); label="sin(x)", legend=:bottomright)
scatter!(x, uₙ; label="average grid point")

# print(norm(uₙ - sin.(x),Inf))
# norm(uₙ - sin.(x),1)
```

Comparing the error to the midpoint method, we see that the errors are very similar:

```julia
function average_err(u, c, f, n)
    x = range(0,1;length=n)
    h=step(x)
    A = Bidiagonal([1; fill(1/h, n-1)], fill(-1/h, n-1), :L)
    ua = A\[c; (f.(x[1:end-1]) + f.(x[2:end]))/2]
    norm(ua - u.(x), Inf)
end

c = 0 # u(0) = 0
f = x -> cos(x)
m = (x[1:end-1] + x[2:end])/2 # midpoints
ns = 10 .^ (1:8) # solve up to n = 10 million


scatter(ns, mid_err.(sin, 0, f, ns); xscale=:log10, yscale=:log10, label="mid")
scatter!(ns, average_err.(sin, 0, f, ns); label="average")
plot!(ns, ns .^ (-1); label="1/n")
plot!(ns, ns .^ (-2); label="1/n^2")
```

```julia
print(mid_err.(sin, 0, f, ns) - average_err.(sin, 0, f, ns))
```

Now looking at the $L_1$ norm, we see it is converging, but to a smaller error before it starts to increase:

```julia
function average_err_l1(u, c, f, n)
    x = range(0,1;length=n)
    h=step(x)
    A = Bidiagonal([1; fill(1/h, n-1)], fill(-1/h, n-1), :L)
    ua = A\[c; (f.(x[1:end-1]) + f.(x[2:end]))/2]
    norm(ua - u.(x), 1)
end

scatter(ns, average_err_l1.(sin, 0, f, ns); xscale=:log10, yscale=:log10, label="L_1")
scatter!(ns, average_err.(sin, 0, f, ns); label="Inf")
plot!(ns, ns .^ (-1); label="1/n")
plot!(ns, ns .^ (-2); label="1/n^2")
```

## 3. Euler methods

**Problem 3.1 (B)** Solve the following ODEs 
using forward and/or backward Euler and increasing $n$, the number of time-steps, 
until $u(1)$ is determined to 3 digits:
$$
\begin{align*}
u(0) &= 1, u'(t) = \cos(t) u(t) + t \\
v(0) &= 1, v'(0) = 0, v''(t) = \cos(t) v(t) + t \\
w(0) &= 1, w'(0) = 0, w''(t) = t w(t) + 2 w(t)^2
\end{align*}
$$
If we increase the initial condition $w(0) = c > 1$, $w'(0)$
the solution may blow up in finite time. Find the smallest positive integer $c$
such that the numerical approximation suggests the equation
does not blow up.

**SOLUTION**
```julia
function first_eq(n)
    #this function takes n and returns the estimate of u(1) using n steps
    #define the range of t
    t = range(0, 1; length=n)
    #find the step-size h
    h = step(t)

    #preallocate memory
    u = zeros(n)
    #set initial condition
    u[1] = 1
    for k=1:n-1
       u[k+1] = (1+h*cos(t[k]))*u[k] + h*t[k] 
    end
    u[end]
end
ns = 2 .^ (1:13)
println(first_eq.(ns)')
```

We see that $u(1) = 2.96$ to three digits.

```julia
#define A(t)
A = t -> [0 1; cos(t) 0]

function second_eq(n)
    #this function takes n and returns the estimate of v(1) using n steps
    #define the range of t
    t = range(0, 1; length=n)
    #find the step-size h
    h = step(t)
    
    #preallocate memory
    u = zeros(2,n)
    #set initial condition
    u[:,1] = [1.0, 0.0]
    for k=1:n-1
       u[:,k+1] = (I + h .* A(t[k]))*u[:,k] + h .* [0, t[k]] 
    end
    u[1,end]
end
ns = 2 .^ (1:13)
println(second_eq.(ns)')
```
We see that $v(1)$ is 1.66 to three digits. Finally,
```julia
#define F(t)
function F(t, w)
   [w[2], t*w[1] + 2*w[1]*w[1]]
end

function third_eq(n=1000, c=1.0)
    #this function takes n and returns the estimate of w(1)
    #using n steps and with initial condition w(0) = c, with c defaulting to 0
    #if no argument is passed
    
    #define the range of t
    t = range(0, 1; length=n)
    #find the step-size h
    h = step(t)
    #preallocate memory
    w = zeros(2,n)
    #set initial condition
    w[:,1] = [c, 0.0]
    for k=1:n-1
       w[:,k+1] = w[:,k] + h .* F(t[k], w[:,k])
    end
    w[1,end]
end
ns = 2 .^ (1:18)
println(third_eq.(ns)')
```
For $c = 1$, we see that $w(1) = 2.80$ to 3 digits. Now consider for $c > 1$:
```julia
function third_eq(c)
    #this function takes n and returns the estimate of w(1)
    #using n steps and with initial condition w(0) = c, with c defaulting to 0
    #if no argument is passed
    n=100000
    #define the range of t
    t = range(0, 1; length=n)
    #find the step-size h
    h = step(t)
    #preallocate memory
    w = zeros(2,n)
    #set initial condition
    w[:,1] = [c, 0.0]
    for k=1:n-1
       w[:,k+1] = w[:,k] + h .* F(t[k], w[:,k])
    end
    w[1,end]
end
cs = 2:10
c_vals = third_eq.(cs)
```

It appears that $c = 2$ is the smallest positive integer greater than 1 for which the numerical approximation suggests the equation does not blow up.

**Problem 3.2⋆ (B)** For an evenly spaced grid $t_1, …, t_n$, use the approximation
$$
{u'(t_{k+1}) + u'(t_k) \over 2} ≈ {u_{k+1} - u_k \over h}
$$
to recast
$$
\begin{align*}
u(0) &= c \\
u'(t) &= a(t) u(t) + f(t)
\end{align*}
$$
as a lower bidiagonal linear system. Use forward-substitution to extend this to vector linear problems:
$$
\begin{align*}
𝐮(0) &= 𝐜 \\
𝐮'(t) &= A(t) 𝐮(t) + 𝐟(t)
\end{align*}
$$

**SOLUTION**

We have,
\begin{align*}
\frac{u_{k+1} - u_k}{h} \approx \frac{u'(t_{k+1}) + u'(t_k)}{2} = \frac{a(t_{k+1})u_{k+1} + a(t_{k})u_{k}}{2} + \frac{1}{2}(f(t_{k+1}) + f(t_k)),
\end{align*}
so we can write,
\begin{align*}
\left(\frac{1}{h} - \frac{a(t_{k+1})}{2}\right)u_{k+1} + \left(-\frac{1}{h} - \frac{a(t_{k})}{2}\right)u_k = \frac{1}{2}(f(t_{k+1}) + f(t_k)).
\end{align*}
With the initial condition $u(0) = c$, we can write the whole system as,
$$
\left[\begin{matrix}
1 \\
-\frac{1}{h} - \frac{a(t_1)}{2} && \frac{1}{h} - \frac{a(t_2)}{2} \\
& \ddots && \ddots \\
 & & -\frac{1}{h} - \frac{a(t_{n-1})}{2} && \frac{1}{h} - \frac{a(t_n)}{2}
\end{matrix}\right]\mathbf{u} =  \left[\begin{matrix} 
c \\
\frac{1}{2}\left(f(t_1) + f(t_2)\right) \\
\vdots \\
\frac{1}{2}\left(f(t_{n-1}) + f(t_n)\right)
\end{matrix}\right],
$$
which is lower bidiagonal.

Now if we wish to use forward substitution in a vector linear problem, we can derive in much the same way as above:
$$
\left(\frac{1}{h}I - \frac{A(t_{k+1})}{2}\right)\mathbf{u}_{k+1} + \left(-\frac{1}{h}I - \frac{A(t_{k})}{2}\right)\mathbf{u}_k = \frac{1}{2}(\mathbf{f}(t_{k+1}) + \mathbf{f}(t_k)),
$$
to make the update equation,
$$
\mathbf{u}_{k+1} = \left(I - \frac{h}{2}A(t_{k+1})\right)^{-1}\left(\left(I + \frac{h}{2}A(t_{k})\right)\mathbf{u}_k + \frac{h}{2}(\mathbf{f}(t_{k+1}) + \mathbf{f}(t_k)) \right),
$$
with initial value,
$$
\mathbf{u}_1 = \mathbf{c}.
$$

**Problem 3.3 (B)** Implement the method designed in Problem 3.1 for the negative time Airy equation 
$$
u(0) = 1, u'(0) = 0, u''(t) = -t u(t)
$$
on $[0,50]$.
How many time-steps are needed to get convergence to 1% accuracy (the "eyeball norm")?

**SOLUTION**
We will work with,
$
\mathbf{u}(t) = \left[\begin{matrix}
u(t) \\ u'(t)
\end{matrix} \right],
$
so that our differential equation is:
$$
\mathbf{u}'(t) = \left[\begin{matrix}
u'(t) \\ u''(t)
\end{matrix} \right] =
\left[\begin{matrix}
0 & 1 \\ -t & 0
\end{matrix} \right] \mathbf{u}(t),
$$
so that,
$$
A(t) = \left[\begin{matrix}
0 & 1 \\ -t & 0
\end{matrix} \right],
$$
and with initial conditions,
$$
\mathbf{u}(0) = \left[\begin{matrix}
1 \\ 0
\end{matrix} \right].
$$

We will use the method described in Problem 3.1, with $\mathbf{f}(t) = \mathbf{0}$:

$$
\mathbf{u}_1 = \left[\begin{matrix}
1 \\ 0
\end{matrix} \right], \hspace{5mm}
\mathbf{u}_{k+1} = \left(I - \frac{h}{2}A(t_{k+1})\right)^{-1}\left(I + \frac{h}{2}A(t_{k})\right)\mathbf{u}_k.
$$
```julia
using SpecialFunctions
n = 1000
#define the range of t
t = range(0, 50; length=n)
#find the step-size h
h = step(t)
#define the function a
a = t -> [0 1; -t 0]

#initialise storage vector and set initial conditions
U=zeros(2, n)
U[:,1] = [1.0, 0.0]

#now iterate forward
for k = 1:n-1
    U[:,k+1] = (I - h/2 .* a(t[k+1])) \ ((I + h/2 .* a(t[k])) * U[:,k])
end

#solution found on wolfram alpha
u = t -> real(1/2 * 3^(1/6) * gamma(2/3) * (sqrt(3)airyai((-1 + 0im)^(1/3)*t) + airybi((-1+0im)^(1/3)*t)))

plot(t, u.(t), label="Airy function")
scatter!(t, U[1,:], label="uf", legend=:bottomright, markersize=2)
```

To see when the error goes below 1%, consider the below:

```julia
n = 2 .^(7:14)
function relative_err(n)
    t = range(0, 50; length=n)
    #find the step-size h
    h = step(t)
    #initialise storage vector and set initial conditions
    U=zeros(2, n)
    U[:,1] = [1.0, 0.0]
    #now iterate forward
    for k = 1:n-1
        U[:,k+1] = (I - h/2 .* a(t[k+1])) \ ((I + h/2 .* a(t[k])) * U[:,k])
    end
    norm(U[1,:] - u.(t), Inf)/norm(u.(t), Inf)
end

plot(n, relative_err.(n), xscale=:log10)
plot!([0.01], seriestype=:hline)
```


**Problem 3.4 (A)** Implement Heat on a graph with $m = 50$ nodes with no forcing
and initial condition $u_{m/2}(0) = 1$ and $u_k(0) = 0$, but where the first and last node are fixed
to  zero, that is replace the first and last rows of the differential equation with
the conditions:
$$
u_1(t) = u_m(t) = 0.
$$
Find the time $t$ such that  $\|𝐮(t)\|_∞ <10^{-3}$ to 2 digits.
Hint: Differentiate to recast the conditions as a differential equation.
Vary $n$, the number of time-steps used in Forward Euler, and increase $T$ in the interval $[0,T]$
until the answer doesn't change.
Do a for loop over the time-slices to find the first that satisfies the condition.
(You may wish to call `println` to print the answer and `break` to exit the for loop).

**SOLUTION**

Following the hint, we will begin by writing a function called ```heat_dissipation(T)```, which runs a simulation of the heat equation with the specified conditions up to time $T$. If the condition $\|\mathbf{u}(t)\|_∞ < 10^{-3}$ is met at a time $t^* < T$, then it will return $t^*$, else it will return $T$. We choose the value $n=1000$ not too large so that we can run this on a large range of values for $T$. Also note that we use Backward Euler, which is more stable for smaller values of $n$; $T$ can potentially be quite large, so Forward Euler may be unstable for even moderately large values of $n$.

```julia
function heat_dissipation(T)
    n=1000
    t = range(0, T; length=n)
    m=50
    #find the step-size h
    h = step(t)
    #define the matrix
    Δ = Tridiagonal([fill(1.0, m-2); 0], [0; fill(-2.0, m-2); 0], [0; fill(1.0, m-2)])
    
    #set initial conditions
    u = zeros(m,n)
    u[Int(m/2), 1] = 1
    for k = 1:n-1
        u[:,k+1] = (I - h*Δ)\u[:,k]
        u_inf = norm(u[:,k+1], Inf)
        if(u_inf < 0.001)
           return t[k+1] 
        end
    end
    return t[n]
end
```

We run this on a large range of values for $T$. The function returns approximately constant ($\approx 905$) values when $T > 905$, so this suggests that our answer lies somewhere around 905.

```julia
Ts = 10:10:1000
ts = heat_dissipation.(Ts)
```

Plotting, we can clearly see that the time output by the function becomes the same towards the end of our range, so we will restrict our search to the end of this range.

```julia
plot(Ts, ts, label = "Time threshold reached", legend=:bottomright)
```

Zooming in:

```julia
Ts = range(900, 910,20)
ts = heat_dissipation.(Ts)
plot(Ts,ts)
```
This looks promising, but it seems like the time-output is somewhat unstable even after $T$ is large enough. Inspecting the actual values of the output, we see that this is likely due to the step size we are using - it will be different for different values of $T$ (as $h = \frac{T}{n}$), and so the smallest $t$ in the discretise range may jump up and down if $n$ is not large enough. To be sure of the answer to 2 decimal places, we will need $n$ to be larger than $2 \frac{T}{0.01} \approx 180000$. We will redefine our function with $n = 200000$, and run it on a few different values of $T$ (that are definitely larger than our target time) to be sure we get the same answer to 2 decimal places.

```julia
function heat_dissipation_large_n(T)
    n=200000
    t = range(0, T; length=n)
    m=50
    #find the step-size h
    h = step(t)
    #define the matrix
    Δ = Tridiagonal([fill(1.0, m-2); 0], [0; fill(-2.0, m-2); 0], [0; fill(1.0, m-2)])
    
    #set initial conditions
    u = zeros(m,n)
    u[Int(m/2), 1] = 1
    for k = 1:n-1
        u[:,k+1] = (I - h*Δ)\u[:,k]
        u_inf = norm(u[:,k+1], Inf)
        if(u_inf < 0.001)
           return t[k+1] 
        end
    end
    return t[n]
end

Ts = [903, 904, 905, 906, 907]
ts = heat_dissipation_large_n.(Ts)
```

We can see that each time we get 902.38 to 2 decimal places, so this is our answer.


**Problem 3.5 (B)** Consider the equation
$$
u(0) = 1, u'(t) = -10u(t)
$$
What behaviour do you observe on $[0,10]$ of forward, backward, and that of Problem 3.1
with a step-size of 0.5? What happens if you decrease the step-size to $0.01$? (Hint: you may wish to do a plot and scale the $y$-axis logarithmically,)

**SOLUTION**

```julia
h = 0.5
t = range(0, 10; step=h)
n = length(t)
uᶠ = zeros(n)
uᵇ = zeros(n)
uᵗ = zeros(n)
uᶠ[1] = uᵇ[1] = uᵗ[1] = 1
a = -10
for k = 1:n-1
    uᶠ[k+1] = (1+h*a) * uᶠ[k]
    uᵇ[k+1] = (1-h*a) \ uᵇ[k]
    uᵗ[k+1] = (1-h*a/2) \ (1 + h*a/2) * uᵗ[k]
end

plot(t, abs.(uᶠ); yscale=:log10)
plot!(t, abs.(uᵇ); yscale=:log10)
plot!(t, abs.(uᵗ); yscale=:log10)
```
We observe that for the stepsize $h=0.5$, the forward method blows up while the other methods appear to converge.

```julia
h = 0.01
t = range(0, 10; step=h)
n = length(t)
uᶠ = zeros(n)
uᵇ = zeros(n)
uᵗ = zeros(n)
uᶠ[1] = uᵇ[1] = uᵗ[1] = 1
for k = 1:n-1
    uᶠ[k+1] = (1+h*a) * uᶠ[k]
    uᵇ[k+1] = (1-h*a) \ uᵇ[k]
    uᵗ[k+1] = (1-h*a/2) \ (1 + h*a/2) * uᵗ[k]
end

nanabs(x) = iszero(x) ? NaN : abs(x)

plot(t, nanabs.(uᶠ); yscale=:log10)
plot!(t, nanabs.(uᵇ); yscale=:log10)
plot!(t, nanabs.(uᵗ); yscale=:log10)
```

For a smaller stepsize ($h = 0.01$), the forward method is also able to converge.

## 1. Two-point boundary value problems

**Problem 1.1 (C)** Construct a finite-difference approximation to the
forced Helmholtz equation
$$
\begin{align*}
u(0) &= 0 \\
u(1) &= 0 \\
u'' + k^2 u &= {\rm e}^x
\end{align*}
$$
and find an $n$ such  the error is less than $10^{-4}$ when compared
with the true solution for $k=10$:
$$
u(x) = (-\cos(k x) + {\rm e}^x \cos(k x)^2 + \cot(k) \sin(k x) - {\rm e} \cos(k) \cot(k) \sin(k x) - {\rm e} \sin(k) \sin(k x) + {\rm e}^x \sin(k x)^2)/(1 + k^2)
$$

```julia
function helm(k, n)
    x = range(0, 1; length = n)
    h = step(x)
    # TODO: Create a SymTridiagonal discretisation
    T = SymTridiagonal(ones(n-2)*(-2/h^2 + k^2),ones(n-3)*1/h^2)
    u = T \ exp.(x[2:end-1])
    [0; u; 0]
end

k = 10
u = x -> (-cos(k*x) + exp(x)cos(k*x)^2 + cot(k)sin(k*x) - ℯ*cos(k)cot(k)sin(k*x) - ℯ*sin(k)sin(k*x) + exp(x)sin(k*x)^2)/(1 + k^2)

n = 2048 # TODO: choose n to get convergence
x = range(0, 1; length=n)
@test norm(helm(k, n) - u.(x)) ≤ 1E-4
```


**Problem 1.2 (A)** Discretisations can also be used to solve eigenvalue problems.
Consider the Schrödinger equation with quadratic oscillator:
$$
u(-L) = u(L) = 0, -u'' + x^2 u = λ u
$$
(a) Use the finite-difference approximation to discretise this equation as eigenvalues of a
matrix. Hint: write
$$
\begin{align*}
u(-L) = 0 \\
-u'' + x^2 u - λu = 0\\
u(L) = 0
\end{align*}
$$
and discretise as before, doing row eliminations to arrive at a symmetric tridiagonal
matrix eigenvalue problem. 
(b) Approximate the eigenvalues using `eigvals(A)` (which returns the eigenvalues of a
matrix `A`) with $L = 10$. 
Can you conjecture their exact value if $L = ∞$? Hint: they are integers and the eigenvalues
closest to zero are most accurate.

**SOLUTION**
We discretise on a grid $u_1,u_2,…,u_n$ for an evenly spaced grid between $[-L,L]$, with
step size $h = 2L/(n-1)$. That is, we have the equations:
$$
\begin{bmatrix}
1 \\
-1/h^2 & 2/h^2 + x_2^2  - λ & -1/h^2 \\
    & ⋱ & ⋱ & ⋱ \\
    && -1/h^2 &  2/h^2 + x_{n-1}^2  - λ & -1/h^2 \\
    &&&& 1 \end{bmatrix} 
    \begin{bmatrix} u_1 \\ \vdots \\ u_n \end{bmatrix} = 0
$$
Row eliminations at the top and bottom reduce this equation to:
$$
\begin{bmatrix}
 2/h^2 + x_2^2   & -1/h^2 \\
    & ⋱ & ⋱ & ⋱ \\
    && -1/h^2 &  2/h^2 + x_{n-1}^2   \end{bmatrix} 
    \begin{bmatrix} u_2 \\ \vdots \\ u_{n-1} \end{bmatrix} = λ\begin{bmatrix} u_2 \\ \vdots \\ u_{n-1} \end{bmatrix} 
$$
This is a standard eigenvalue problem and we can compute the eigenvalues using `eigvals`:
```julia
L = 10
n = 1000
x = range(-L,L; length=n)
h = step(x)
eigvals(SymTridiagonal(fill(2/h^2,n-2)  + x[2:end-1].^2, fill(-1/h^2, n-3)))
```

On inspection of the smallest values, it seems that the positive odd integers are the eigenvalues for $L = \infty$. Increasing $L$ (and also $n$) it becomes more obvious:

```julia
L = 100
n = 10000
x = range(-L,L; length = n)
h = step(x)
A = SymTridiagonal(x[2:end-1] .^ 2 .+ 2/h^2,ones(n-3)* (-1)/h^2)
sort((eigvals(A)))[1:20]
```


**Problem 1.3⋆ (A)** Consider Helmholtz with Neumann conditions:
$$
u'(0) = c_0 \\
u'(1) = c_1 \\
u_{xx} + k^2 u = f(x)
$$
Write down the finite difference approximation approximating
$u(x_k) ≈ u_k$ on
 an evenly spaced grid $x_k = (k-1)/(n-1)$ for $k=1,…,n$
using the first order derivative approximation conditions:
$$
\begin{align*}
u'(0) &≈ (u_2-u_1)/h = c_0 \\
u'(1) &≈ (u_n-u_{n-1})/h  = c_1
\end{align*}
$$
Use pivoting to reduce the equation to one involving a
symmetric tridiagonal matrix.

**SOLUTION**

We have, with $u(x_k) = u_k$ (and using $\kappa$ instead of $k$ in the equation $u_{xx} + k^2u = f(x)$ so as to avoid confusion with the indices):
\begin{align*}
\frac{u_2 - u_1}{h} &= c_0, \\
\frac{u_{k-1} - 2u_k + u_{k+1}}{h^2} + \kappa^2u_k &= f(x_k), \hspace{5mm} \textrm{ for } k=2:n-1\\
\frac{u_n - u_{n-1}}{h} &= c_1, 
\end{align*}
which we write in matrix form as:

$$
\left[\begin{matrix}
-\frac{1}{h} & \frac{1}{h} \\
\frac{1}{h^2} & \kappa^2 - \frac{2}{h^2} & \frac{1}{h^2} \\
&\ddots & \ddots & \ddots \\
&&\frac{1}{h^2} & \kappa^2 - \frac{2}{h^2} & \frac{1}{h^2} \\
&&& -\frac{1}{h} & \frac{1}{h}
\end{matrix}
\right] \mathbf{u} = \left[\begin{matrix}
c_0 \\ f(x_2)\\ \vdots \\f(x_{n-1}) \\ c_1
\end{matrix}\right],
$$
which we can make symmetric tridiagonal by multiplying the first row by $1/h$ and the final row by $-1/h$:
$$
\left[\begin{matrix}
-\frac{1}{h^2} & \frac{1}{h^2} \\
\frac{1}{h^2} & \kappa^2 - \frac{2}{h^2} & \frac{1}{h^2} \\
&\ddots & \ddots & \ddots \\
&&\frac{1}{h^2} & \kappa^2 - \frac{2}{h^2} & \frac{1}{h^2} \\
&&& \frac{1}{h^2} & -\frac{1}{h^2}
\end{matrix}
\right] \mathbf{u} = \left[\begin{matrix}
\frac{c_0}{h} \\ f(x_2)\\ \vdots \\f(x_{n-1}) \\ -\frac{c_1}{h}
\end{matrix}\right],
$$