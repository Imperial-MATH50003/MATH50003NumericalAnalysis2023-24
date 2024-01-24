# # MATH50003 (2022–23)
# # Lab 5: III.1 Structured Matrices and III.2 Differential Equations


# **Remark** One should normally not need to implement these methods oneselves as there
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
# different behaviour, so it is important to understand what is happening.


We have seen how algebraic operations (`+`, `-`, `*`, `/`) are
defined exactly in terms of rounding ($⊕$, $⊖$, $⊗$, $⊘$) 
for floating point numbers. Now we see how this allows us
to do (approximate) linear algebra operations on matrices. 

A matrix can be stored in different formats. Here we consider the following structures:


1. _Dense_: This can be considered unstructured, where we need to store all entries in a
vector or matrix. Matrix multiplication reduces directly to standard algebraic operations. 
Solving linear systems with dense matrices will be discussed later.
2. _Triangular_: If a matrix is upper or lower triangular, we can immediately invert using
back-substitution. In practice we store a dense matrix and ignore the upper/lower entries.
3. _Banded_: If a matrix is zero apart from entries a fixed distance from  the diagonal it is
called banded and this allows for more efficient algorithms. We discuss diagonal, 
tridiagonal and bidiagonal matrices.

In the next chapter we consider more complicated orthogonal matrices.


```julia
# LinearAlgebra contains routines for doing linear algebra
# BenchmarkTools is a package for reliable timing
using LinearAlgebra, Plots, BenchmarkTools, Test
```

-----

## 1. Dense vectors and matrices

A `Vector` of a primitive type (like `Int` or `Float64`) is stored
consecutively in memory: that is, a vector consists of a memory address (a _pointer_)
to the first entry and a length. E.g. if we have a `Vector{Int8}` of length
`n` then it is stored as `8n` bits (`n` bytes) in a row.
That is, if the memory address of the first entry is `k` and the type
is `T`, the memory
address of the second entry is `k + sizeof(T)`. 

--------

**Remark (advanced)** We can actually experiment with this
(NEVER DO THIS IN PRACTICE!!), beginning with an 8-bit type:
```julia
a = Int8[2, 4, 5]
p = pointer(a) # pointer(a) returns memory address of the first entry, which is the displayed hex number
# We can think of a pointer as simply a UInt64 alongside a Type to interpret what is stored
```
We can see what's stored at a pointer as follows:
```julia
Base.unsafe_load(p) # loads data at `p`. Knows its an `Int8` because of type of `Ptr`
```
Adding an integer to a pointer gives a new pointer with the address incremented:
```julia
p + 1 # memory address of next entry, which is 1 more than first
```
We see that this gives us the next entry:
```julia
Base.unsafe_load(p) # loads data at `p+1`, which is second entry of the vector
```
For other types we need to increment the address by the size of the type:
```julia
a = [2.0, 1.3, 1.4]
p = pointer(a)
Base.unsafe_load(p + 8) # sizeof(Float64) == 8
```
Why not do this in practice? It's unsafe because there's nothing stopping us from going past the end of an array:
```julia
Base.unsafe_load(p + 3 * 8) # whatever bits happened to be next in memory, usually nonsense
```
This may even crash Julia! (I got lucky that it didn't when producing the notes.)

------


A  `Matrix` is stored consecutively in memory, going down column-by-
column (_column-major_). That is,
```julia
A = [1 2;
     3 4;
     5 6]
```
Is actually stored equivalently to a length `6` vector:
```julia
vec(A)
```
which in this case would be stored using in `8 * 6 = 48` consecutive
memory addresses. That is, a matrix is a pointer to the first entry alongside two integers
dictating the row and column sizes.

-----

**Remark (advanced)** Note that transposing `A` is done lazyily 
and so `transpose(A)` (which is equivalent to the adjoint/conjugate-transpose
`A'` when the entries are real),
is just a special type with a single field: `transpose(A).parent == A`.
This is equivalent to 
_row-major_ format, where the next address in memory of `transpose(A)` corresponds to
moving along the row.

-----

Matrix-vector multiplication works as expected:
```julia
x = [7, 8]
A * x
```

Note there are two ways this can be implemented: 

**Algorithm 1 (matrix-vector multiplication by rows)**
For a ring $R$ (typically $ℝ$ or $ℂ$), $A ∈ R^{m × n}$ and $𝐱 ∈ R^n$ we have
$$
A𝐱 = \begin{bmatrix} ∑_{j=1}^n a_{1,j} x_j \\ ⋮ \\ ∑_{j=1}^n a_{m,j} x_j \end{bmatrix}.
$$
In code this can be implemented for any types that support `*` and `+` as follows:
```julia
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
```

**Algorithm 2 (matrix-vector multiplication by columns)**
For a ring $R$ (typically $ℝ$ or $ℂ$), $A ∈ R^{m × n}$ and $𝐱 ∈ R^n$ we have
$$
A 𝐱 = x_1 𝐚_1  + ⋯ + x_n 𝐚_n
$$
where $𝐚_j := A 𝐞_j ∈ R^m$ (that is, the $j$-th column of $A$). In code this can be implemented for any types that support `*` and `+` 
as follows:
```julia
function mul_cols(A, x)
    m,n = size(A)
    # promote_type type finds a type that is compatible with both types, eltype gives the type of the elements of a vector / matrix
    T = promote_type(eltype(x),eltype(A))
    c = zeros(T, m) # the returned vector, begins of all zeros
    for j = 1:n, k = 1:m
        c[k] += A[k, j] * x[j] # equivalent to c[k] = c[k] + A[k, j] * x[j]
    end
    c
end
```

Both implementations match exactly for integer inputs:
```julia
mul_rows(A, x), mul_cols(A, x) # also matches `A*x`
```


Either implementation will be $O(mn)$ operations. However, the implementation 
`mul_cols` accesses the entries of `A` going down the column,
which happens to be _significantly faster_ than `mul_rows`, due to accessing
memory of `A` in order. We can see this by measuring the time it takes using `@btime`:
```julia
n = 1000
A = randn(n,n) # create n x n matrix with random normal entries
x = randn(n) # create length n vector with random normal entries

@btime mul_rows(A,x)
@btime mul_cols(A,x)
@btime A*x; # built-in, high performance implementation. USE THIS in practice
```
Here `ms` means milliseconds (`0.001 = 10^(-3)` seconds) and `μs` means microseconds (`0.000001 = 10^(-6)` seconds).
So we observe that `mul` is roughly 3x faster than `mul_rows`, while the optimised `*` is roughly 5x faster than `mul`.

-----

**Remark (advanced)** For floating point types, `A*x` is implemented in BLAS which is generally multi-threaded
and is not identical to `mul_cols(A,x)`, that is, some inputs will differ in how the computations
are rounded.

-----


Note that the rules of floating point arithmetic apply here: matrix multiplication with floats
will incur round-off error (the precise details of which are subject to the implementation):

```julia
A = [1.4 0.4;
     2.0 1/2]
A * [1, -1] # First entry has round-off error, but 2nd entry is exact
```
And integer arithmetic will be subject to overflow:
```julia
A = fill(Int8(2^6), 2, 2) # make a matrix whose entries are all equal to 2^6
A * Int8[1,1] # we have overflowed and get a negative number -2^7
```


Solving a linear system is done using `\`:
```julia
A = [1 2 3;
     1 2 4;
     3 7 8]
b = [10; 11; 12]
A \ b
```
Despite the answer being integer-valued, 
here we see that it resorted to using floating point arithmetic,
incurring rounding error. 
But it is "accurate to (roughly) 16-digits".
As we shall see, the way solving a linear system works is we first write `A` as a
product of matrices that are easy to invert, e.g., a product of triangular matrices or a product of an orthogonal
and triangular matrix.


## 2. Triangular matrices

Triangular matrices are represented by dense square matrices where the entries below the
diagonal
are ignored:
```julia
A = [1 2 3;
     4 5 6;
     7 8 9]
U = UpperTriangular(A)
```
We can see that `U` is storing all the entries of `A` in a field called `data`:
```julia
U.data
```
Similarly we can create a lower triangular matrix by ignoring the entries above the diagonal:
```julia
L = LowerTriangular(A)
```

If we know a matrix is triangular we can do matrix-vector multiplication in roughly half
the number of operations by skipping over the entries we know are zero:

**Algorithm 3 (upper-triangular matrix-vector multiplication by columns)**
```julia
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
```


Moreover, we can easily invert matrices. 
Consider a simple 3×3 example, which can be solved with `\`:
```julia
b = [5, 6, 7]
x = U \ b # Excercise: why does this return a float vector?
```
Behind the seens, `\` is doing back-substitution: considering the last row, we have all
zeros apart from the last column so we know that `x[3]` must be equal to:
```julia
b[3] / U[3,3]
```
Once we know `x[3]`, the second row states `U[2,2]*x[2] + U[2,3]*x[3] == b[2]`, rearranging
we get that `x[2]` must be:
```julia
(b[2] - U[2,3]*x[3])/U[2,2]
```
Finally, the first row states `U[1,1]*x[1] + U[1,2]*x[2] + U[1,3]*x[3] == b[1]` i.e.
`x[1]` is equal to
```julia
(b[1] - U[1,2]*x[2] - U[1,3]*x[3])/U[1,1]
```

More generally, we can solve the upper-triangular system using _back-substitution_:

**Algorithm 4 (back-substitution)** Let $𝔽$ be a field (typically $ℝ$ or $ℂ$).
 Suppose $U ∈ 𝔽^{n × n}$ is upper-triangular
and invertible. Then for $𝐛 ∈ 𝔽^n$ the solution $𝐱 ∈ 𝔽^n$ to $U 𝐱 = 𝐛$, that is,
$$
\begin{bmatrix}
u_{11} & ⋯ & u_{1n} \\ & ⋱ & ⋮ \\ && u_{nn}
\end{bmatrix} \begin{bmatrix} x_1 \\ ⋮ \\ x_n \end{bmatrix} = 
\begin{bmatrix} b_1 \\ ⋮ \\ b_n \end{bmatrix}
$$
is given by computing $x_n, x_{n-1}, …, x_1$ via:
$$
x_k = {b_k - ∑_{j=k+1}^n u_{kj} x_j \over u_{kk}}
$$
In code this can be implemented for any types that support `*`, `+` and `/` as follows:
```julia
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
```

The problem sheet will explore implementing multiplication and forward substitution 
for lower triangular matrices. 
The cost of multiplying and solving linear systems with a
triangular matrix is $O(n^2)$.

------

## 3. Banded matrices

A _banded matrix_ is zero off a prescribed number of diagonals. 
We call the number of (potentially) non-zero diagonals the _bandwidths_:


**Definition 1 (bandwidths)** A matrix $A$ has _lower-bandwidth_ $l$ if 
$A[k,j] = 0$ for all $k-j > l$ and _upper-bandwidth_ $u$ if
$A[k,j] = 0$ for all $j-k > u$. We say that it has _strictly lower-bandwidth_ $l$
if it has lower-bandwidth $l$ and there exists a $j$ such that $A[j+l,j] \neq 0$.
We say that it has _strictly upper-bandwidth_ $u$
if it has upper-bandwidth $u$ and there exists a $k$ such that $A[k,k+u] \neq 0$.


### Diagonal

**Definition 2 (Diagonal)** _Diagonal matrices_ are square matrices with bandwidths $l = u = 0$.


Diagonal matrices in Julia are stored as a vector containing the diagonal entries:
```julia
x = [1,2,3]
D = Diagonal(x) # the type Diagonal has a single field: D.diag
```
It is clear that we can perform diagonal-vector multiplications and solve linear systems involving diagonal matrices efficiently
(in $O(n)$ operations).

### Bidiagonal

**Definition 3 (Bidiagonal)** If a square matrix has bandwidths $(l,u) = (1,0)$ it is _lower-bidiagonal_ and
if it has bandwidths $(l,u) = (0,1)$ it is _upper-bidiagonal_. 

We can create Bidiagonal matrices in Julia by specifying the diagonal and off-diagonal:

```julia
L = Bidiagonal([1,2,3], [4,5], :L) # the type Bidiagonal has three fields: L.dv (diagonal), L.ev (lower-diagonal), L.uplo (either 'L', 'U')
```
```julia
Bidiagonal([1,2,3], [4,5], :U)
```

Multiplication and solving linear systems with Bidiagonal systems is also $O(n)$ operations, using the standard
multiplications/back-substitution algorithms but being careful in the loops to only access the non-zero entries. 


### Tridiagonal

**Definition 4 (Tridiagonal)** If a square matrix has bandwidths $l = u = 1$ it is _tridiagonal_.

Julia has a type `Tridiagonal` for representing a tridiagonal matrix from its sub-diagonal, diagonal, and super-diagonal:
```julia
T = Tridiagonal([1,2], [3,4,5], [6,7]) # The type Tridiagonal has three fields: T.dl (sub), T.d (diag), T.du (super)
```
Tridiagonal matrices will come up in solving second-order differential equations and orthogonal polynomials.
We will later see how linear systems involving tridiagonal matrices can be solved in $O(n)$ operations.


**Example**

Let's do an example of integrating $\cos x$, and see if our method matches
the true answer of $\sin x$. First we construct the system
as a lower-triangular, `Bidiagonal` matrix:
```julia
using LinearAlgebra, Plots

function indefint(x)
    h = step(x) # x[k+1]-x[k]
    n = length(x)
    L = Bidiagonal([1; fill(1/h, n-1)], fill(-1/h, n-1), :L)
end

n = 10
x = range(0, 1; length=n)
L = indefint(x)
```
We can now solve for our particular problem using both the left and 
mid-point rules:
```julia
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
```
They both are close though the mid-point version is significantly
more accurate.
 We can estimate how fast it converges:
```julia
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


