# # MATH50003 (2023–24)
# # Revision Lab


# **Problem 1** Use finite-differences with an appropriately chosen `h`
# to approximate the derivative in Newton iteration. Does the method still converge with an accurate initial guess?



# **Problem 2.2** Create a positive `Float64` whose exponent is $q = 156$ and has significand
# bits
# $$
# b_k = \begin{cases}
#     1 & k\hbox{ is prime} \\
#     0 & \hbox{otherwise}
#     \end{cases}
# $$
# Hint: use the `gcd` function to determine if a number is prime.




# **Problem 2** Use matrix representations to compute derivatives of $f(x) = 1 + x + x^2$ and $1 + x/3 + x^2$
# by hand and in Julia.
# Hint: Use `I` or `one(x)` instead of `1`. 


# **Problem 3** Does a `Dual{Dual{T}}` work for second derivatives?



# **Problem 3.3** Use `Dual` and Newton iteration to find a frequency `ω` such that
# the heat on a graph with 50 nodes is equal to zero at time $T = 0$ at node $25$, using Forward Euler
# with 200 time-steps to approximate the solution to the differential equation.
# (Hint: use `Uᶠ = zeros(typeof(ω), m, n)` to ensure duals are allowed and use an initial guess of
# `ω = 1`.)



# **Problem 1** Complete the implementation of a type representing
# permutation matrices that supports `P[k,j]` in $O(1)$ operations and `*` in $O(n)$ operations,
# where $n$ is the length of the permutation.


struct PermutationMatrix <: AbstractMatrix{Int}
    p::Vector{Int} # represents the permutation whose action is v[p]
    ## This is an internal constructor: allows us to check validity of the input.
    function PermutationMatrix(p::Vector)
        sort(p) == 1:length(p) || error("input is not a valid permutation")
        new(p)
    end
end

function size(P::PermutationMatrix)
    (length(P.p),length(P.p))
end

## getindex(P, k, j) is a synonym for P[k,j]
function getindex(P::PermutationMatrix, k::Int, j::Int)
    ## TODO: Implement P[k,j]
    
end
function *(P::PermutationMatrix, x::AbstractVector)
    ## TODO: return a vector whose entries are permuted according to P.p
    
end

## If your code is correct, this "unit test" will succeed
p = [1, 4, 2, 5, 3]
P = PermutationMatrix(p)
@test P == I(5)[p,:]

n = 100_000
p = Vector(n:-1:1) # makes a Vector corresponding to [n,n-1,…,1]
P = PermutationMatrix(p)
x = randn(n)
@test P*x == x[p]

