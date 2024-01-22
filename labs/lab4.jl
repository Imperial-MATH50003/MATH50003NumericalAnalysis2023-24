# # MATH50003 (2022–23)
# # Lab 4: II.3 Floating Point Arithmetic and II.4 Interval Arithmetic

# This lab explores the usage of rounding modes for floating point arithmetic and how they
# can be used to compute _rigorous_ bounds on mathematical constants such as ℯ.
# The key idea is _interval arithmetic_.
#
# This will be consist of the following:
# 1. The finite Taylor series $\exp x ≈ ∑_{k=0}^n x^k/k!$ where each operation is now
#    an interval operation
# 2. A bound on $∑_{k=n+1}^∞ x^k/k!$ that we capture in the returned result
#

# **Learning Outcomes**
#
# Mathematical knowledge:
#
# 1. Behaviour of floating point rounding and interval arithmetic.
# 2. Combining interval arithmetic with Taylor series bounds for rigorous computations.
#
# Coding knowledge:
#
# 1. Setting the rounding mode in constructors like `Float32` and via `setrounding`.
# 2. High precision floating point numbers via `big` and setting precision via `setprecision`.

# We need the following packages:

using SetRounding, Test

# ## II.3 Floating Point Arithmetic

# In Julia, the rounding mode is specified by tags `RoundUp`, `RoundDown`, and
# `RoundNearest`. (There are also more exotic rounding strategies `RoundToZero`, `RoundNearestTiesAway` and
# `RoundNearestTiesUp` that we won't use.)



# Let's try rounding a `Float64` to a `Float32`.


printlnbits(1/3)  # 64 bits
printbits(Float32(1/3))  # round to nearest 32-bit

# The default rounding mode can be changed:

printbits(Float32(1/3,RoundDown) )

# Or alternatively we can change the rounding mode for a chunk of code
# using `setrounding`. The following computes upper and lower bounds for `/`:

x = 1f0
setrounding(Float32, RoundDown) do
    x/3
end,
setrounding(Float32, RoundUp) do
    x/3
end


# **WARNING (compiled constants, non-examinable)**: Why did we first create a variable `x` instead of typing `1f0/3`?
# This is due to a very subtle issue where the compiler is _too clever for it's own good_: 
# it recognises `1f0/3` can be computed at compile time, but failed to recognise the rounding mode
# was changed. 

# **Problem 1** Complete functions `exp_t_3_down`/`exp_t_3_up` implementing the first
# three terms of the Taylor expansion of $\exp(x)$, that is, $1 + x + x/2 + x^2/6$ but where
# each operation is rounded down/up. Use `typeof(x)` to make sure you are changing the
# rounding mode for the right floating point type.

function exp_t_3_down(x)
    ## TODO: use setrounding to compute 1 + x + x/2 + x^2/6 but rounding down
    
end

function exp_t_3_up(x)
    ## TODO: use setrounding to compute 1 + x + x/2 + x^2/6 but rounding up
    
end

@test exp_t_3_down(Float32(1)) ≡ 2.6666665f0 # ≡ checks type and all bits are equal
@test exp_t_3_up(Float32(1)) ≡ 2.6666667f0



# ### High-precision floating-point numbers


# It is possible to set the precision of a floating-point number
# using the `BigFloat` type, which results from the usage of `big`
# when the result is not an integer.
# For example, here is an approximation of 1/3 accurate
# to 77 decimal digits:

big(1)/3

# Note we can set the rounding mode as in `Float64`, e.g., 
# this gives (rigorous) bounds on
# `1/3`:

setrounding(BigFloat, RoundDown) do
  big(1)/3
end, setrounding(BigFloat, RoundUp) do
  big(1)/3
end

# We can also increase the precision, e.g., this finds bounds on `1/3` accurate to 
# more than 1000 decimal places:

setprecision(4_000) do # 4000 bit precision
  setrounding(BigFloat, RoundDown) do
    big(1)/3
  end, setrounding(BigFloat, RoundUp) do
    big(1)/3
  end
end




# -----
#
# ## II.4 Interval Arithmetic

# 
# We will now create a Type to represent an interval, which we will call `Interval`.
# We need two fields: the left endpoint (`a`) and a right endpoint (`b`):

struct Interval # represents the set {x : a ≤ x ≤ b}
    a
    b
end

# For example, if we say `X = Interval(1, 2)` this corresponds to the mathematical interval
# $[1, 2]$, and the fields are accessed via `X.a` and `X.b`.
# We will overload `*`, `+`, `-`, `/` to use interval arithmetic. That is, whenever we do arithmetic with
# an instance of `Interval` we want it to use correctly rounded interval varients. 
# We also need to support `one` (a function that creates an interval containing a single point `1`)
# and `in` functions (a function to test if a number is within an interval).
# To overload these functions we need to import them as follows:

import Base: *, +, -, /, one, in


# We can overload `one` as follows to create an interval corresponding to $[1,1]$.
# First note that the `one(T)` function will create the "multiplicative identity"
# for a given type. For example `one(Int)` will return `1`, `one(Float64)` returns `1.0`,
# and `one(String)` returns "" (because `"" * "any string" == "any string"`):

one(Int), one(Int64), one(String)

# We can also just call it on an instance of the type:

one(2), one(2.0), one("any string")

# For an interval the multiplicative identity is the interval whose lower and upper limit are both 1.
# To ensure its the right type we call `one(X.a)` and `one(X.b)`

one(X::Interval) = Interval(one(X.a), one(X.b))

# Thus the following returns an interval whose endpoints are both `1.0`:

one(Interval(2.0,3.3))

# Now if `X = Interval(a,b)` this corresponds to the mathematical interval $[a,b]$.
# And a real number $x ∈ [a,b]$ iff $a ≤ x ≤ b$. In Julia the endpoints $a$ and $b$ are accessed
# via $X.a$ and $B.b$ hence the above test becomes `X.a ≤ x ≤ X.b`. Thus we overload `in` 
# as follows:

in(x, X::Interval) = X.a ≤ x ≤ X.b

# The function `in` is whats called an "infix" operation (just like `+`, `-`, `*`, and `/`). We can call it
# either as `in(x, X)` or put the `in` in the middle and write `x in X`. This can be seen in the following:

X = Interval(2.0,3.3)
## 2.5 in X is equivalent to in(2.5, X)
## !(3.4 in X) is equivalent to !in(3.4, X)
2.5 in X, !(3.4 in X)

# The first problem now is to overload arithmetic operations to do the right thing.

# **Problem 2**  Use the formulae from Problem 1 to complete (by replacing the `# TODO: …` comments with code)
#  the following implementation of an 
# `Interval` 
# so that `+`, `-`, and `/` implement $⊕$, $⊖$, and $⊘$ as defined above.




# Hint: Like `in`, `+` is an infix operation, so if `X isa Interval` and `Y isa Interval`
# then the following function will be called when we call `X + Y`.
# We want it to  implement `⊕` as worked out by hand by replacing the `# TODO` with
# the correct interval versions. For example, for the first `# TODO`, we know the lower bound of
# $X + Y$ is $a + c$, where $X = [a,b]$ and $Y = [c,d]$. But in Julia we access the lower bound of $X$ ($a$)
# via `X.a` and the lower bound of $Y$ via `Y.a`.
# Thus just replace the first `#TODO` with `X.a + Y.a`.

# You can ignore the `T = promote_type(...)` line for now: it is simply finding the right type
# to change the rounding mode by finding the "bigger" of the type of `X.a` and `Y.a`. So in the examples below
# `T` will just become `Float64`.


function +(X::Interval, Y::Interval)
    T = promote_type(typeof(X.a), typeof(Y.a))
    a = setrounding(T, RoundDown) do
        ## TODO: lower bound
        
    end
    b = setrounding(T, RoundUp) do
        ## TODO: upper bound
        
    end
    Interval(a, b)
end

## following example was the non-associative example but now we have bounds
@test Interval(1.1,1.1) + Interval(1.2,1.2) + Interval(1.3,1.3) ≡ Interval(3.5999999999999996, 3.6000000000000005)


# The following function is called whenever we divide an interval by an `Integer` (think of `Integer` for now
# a "superset" containing all integer types, e.g. `Int8`, `Int`, `UInt8`, etc.). Again we want it to return the
# set operation ⊘ with correct rounding.
# Be careful about whether `n` is positive or negative, and you may want to test if `n > 0`. To do so, use an

function /(A::Interval, n::Integer)
    T = typeof(A.a)
    if iszero(n)
        error("Dividing by zero not support")
    end
    a = setrounding(T, RoundDown) do
        ## TODO: lower bound
        
    end
    b = setrounding(T, RoundUp) do
        ## TODO: upper bound
        
    end
    Interval(a, b)
end

@test Interval(1.0,2.0)/3 ≡ Interval(0.3333333333333333, 0.6666666666666667)
@test Interval(1.0,2.0)/(-3) ≡ Interval(-0.6666666666666667, -0.3333333333333333)

# Now we need to overload `*` to behave like the operation `⊗` defined above.
# You will also have to test whether multiple conditions are true.
# The notation `COND1 && COND2` returns true if `COND1` and `COND2` are both true.
# The notation `COND1 || COND2` returns true if either `COND1` or `COND2` are true.
# So for example the statement `0 in A || 0 in B` returns `true` if either interval `A`
# or `B` contains `0`.

function *(A::Interval, B::Interval)
    T = promote_type(typeof(A.a), typeof(B.a))
    if 0 in A || 0 in B
        error("Multiplying with intervals containing 0 not supported.")
    end
    if A.a > A.b || B.a > B.b
        error("Empty intervals not supported.")
    end
    a = setrounding(T, RoundDown) do
        ## TODO: lower bound
        
    end
    b = setrounding(T, RoundUp) do
        ## TODO: upper bound
        
    end
    Interval(a, b)
end

@test Interval(1.1, 1.2) * Interval(2.1, 3.1) ≡ Interval(2.31, 3.72)
@test Interval(-1.2, -1.1) * Interval(2.1, 3.1) ≡ Interval(-3.72, -2.31)
@test Interval(1.1, 1.2) * Interval(-3.1, -2.1) ≡ Interval(-3.72, -2.31)
@test Interval(-1.2, -1.1) * Interval(-3.1, -2.1) ≡ Interval(2.31, 3.72)

# -----

# The following function  computes the first `n+1` terms of the Taylor series of $\exp(x)$:
# $$
# \sum_{k=0}^n {x^k \over k!}
# $$
# (similar to the one seen in lectures).

function exp_t(x, n)
    ret = one(x) # 1 of same type as x
    s = one(x)
    for k = 1:n
        s = s/k * x
        ret = ret + s
    end
    ret
end


# **Problem 3(a)** Bound the tail of the Taylor series for ${\rm e}^x$ assuming $|x| ≤ 1$. 
# (Hint: ${\rm e}^x ≤ 3$ for $x ≤ 1$.)
# 

# 
# **Problem 3(b)** Use the bound
# to write a function `exp_bound` which computes ${\rm e}^x$ with rigorous error bounds, that is
# so that when applied to an interval $[a,b]$ it returns an interval that is 
# guaranteed to contain the interval $[{\rm e}^a, {\rm e}^b]$.


function exp_bound(x::Interval, n)
    ## TODO: Return an Interval such that exp(x) is guaranteed to be a subset
    
end

e_int = exp_bound(Interval(1.0,1.0), 20)
@test exp(big(1)) in e_int
@test exp(big(-1)) in exp_bound(Interval(-1.0,-1.0), 20)
@test e_int.b - e_int.a ≤ 1E-13 # we want our bounds to be sharp

# ------
# **Problem 4** Use `big` and `setprecision` to compute ℯ to a 1000 decimal digits with
# rigorous error bounds. 

# Hint: The function `big` will create a `BigFloat` version of a `Float64` and the type
# `BigFloat` allows changing the number of signficand bits. In particular, the code block
# ```julia
# setprecision(NUMSIGBITS) do
#
# end
# ```
# will use the number of significand bits specified by `NUMSIGBITS` for any `BigFloat` created
# between the `do` and the `end`. 

