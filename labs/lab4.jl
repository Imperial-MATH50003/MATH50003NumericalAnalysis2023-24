# # MATH50003 (2023–24)
# # Lab 4: II.3 Floating Point Arithmetic and II.4 Interval Arithmetic

# This lab explores the usage of rounding modes for floating point arithmetic and how they
# can be used to compute _rigorous_ bounds on mathematical constants such as `ℯ`.
# The key idea is using _interval arithmetic_ to compute the Taylor series which is
# combined with a bound on the error caused by truncating a Taylor series.
# As a fun example, we compute the first 1000 digits of `ℯ`, backed up by a rigorous
# computation.

# **Learning Outcomes**
#
# Mathematical knowledge:
#
# 1. Behaviour of floating point rounding and interval arithmetic.
# 2. Extending interval arithmetic operations to non-positive intervals.
# 3. Combining interval arithmetic with Taylor series bounds for rigorous computations.
#
# Coding knowledge:
#
# 1. Setting the rounding mode in constructors like `Float32` and via `setrounding`.
# 2. High precision floating point numbers via `big` and setting precision via `setprecision`.
# 3. The `promote` command for converting multiple variables to be the same type.
# 4. Using `&&` for "and" and `||` for "or".

# We need the following packages:

using ColorBitstring, SetRounding, Test

# ## II.3 Floating Point Arithmetic

# In Julia, the rounding mode is specified by tags `RoundUp`, `RoundDown`, and
# `RoundNearest`. (There are also more exotic rounding strategies `RoundToZero`, `RoundNearestTiesAway` and
# `RoundNearestTiesUp` that we won't use.)



# Let's try rounding a `Float64` to a `Float32`.


printlnbits(1/3)  # 64 bits
printbits(Float32(1/3))  # round to nearest 32-bit

# The default rounding mode can be changed:

printbits(Float32(1/3,RoundDown) ) # Rounds from a Float64 to Float32, rounding down

# Or alternatively we can change the rounding mode for a chunk of code
# using `setrounding`. The following computes upper and lower bounds for `/`:

x = 1f0
setrounding(Float32, RoundDown) do
    x/3
end,
setrounding(Float32, RoundUp) do
    x/3
end


# **WARNING (compiled constants)**: Why did we first create a variable `x` instead of typing `1f0/3`?
# This is due to a very subtle issue where the compiler is _too clever for it's own good_: 
# it recognises `1f0/3` can be computed at compile time, but failed to recognise the rounding mode
# was changed. 

# **Problem 1** Complete functions `exp_t_3_down`/`exp_t_3_up` implementing the first
# three terms of the Taylor expansion of $\exp(x)$, that is, $1 + x + x^2/2 + x^3/6$ but where
# each operation is rounded down/up. Use `typeof(x)` to make sure you are changing the
# rounding mode for the right floating point type.

function exp_t_3_down(x)
    T = typeof(x) # use this to set the rounding mode
    ## TODO: use setrounding to compute 1 + x + x/2 + x^2/6 but rounding down
    
end

function exp_t_3_up(x)
    ## TODO: use setrounding to compute 1 + x + x/2 + x^2/6 but rounding up
    
end

@test exp_t_3_down(Float32(1)) ≡ 2.6666665f0 # ≡ checks type and all bits are equal
@test exp_t_3_up(Float32(1)) ≡ 2.6666667f0



# ### High-precision floating-point numbers


# It is possible to get higher precision (more signficand and exponent bits)
#  of a floating-point number
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


# **Problem 2** Inbuilt functions like `exp`, `sqrt`, etc. support `BigFloat`.
# Compute at least the first thousand decimal digits of `ℯ` using `setprecision`
# and the inbuilt `exp` function.

## TODO: Use big and setprecision to compute the first thousand digits of ℯ.




# -----
#
# ## II.4 Interval Arithmetic

# 
# We will now create a Type to represent an interval $[a,b] = {x : a ≤ x ≤ b}$, which we will call `Interval`.
# We need fields for the left endpoint (`a`) and a right endpoint (`b`):

struct Interval # represents the set [a,b]
    a # left endpoint
    b # right endpoint
end

Interval(x) = Interval(x,x) # Support Interval(1) to represent [1,1]

# For example, if we say `X = Interval(1, 2)` this corresponds to the mathematical interval
# $[1, 2]$, and the fields are accessed via `X.a` and `X.b`.
# We will overload `*`, `+`, `-`, `/` to use interval arithmetic. That is, whenever we do arithmetic with
# an instance of `Interval` we want it to use correctly rounded interval variants. 
# We also need to support `one` (a function that creates an interval containing a single point `1`)
# and `in` functions (a function to test if a number is within an interval).
# To overload these functions we need to import them as follows:

import Base: *, +, -, ^, /, one, in

# We overload `in` as follows:

in(x, X::Interval) = X.a ≤ x ≤ X.b

# The function `in` is whats called an "infix" operation (just like `+`, `-`, `*`, and `/`). We can call it
# either as `in(x, X)` or put the `in` in the middle and write `x in X`. This can be seen in the following:

X = Interval(2.0,3.3)
## 2.5 in X is equivalent to in(2.5, X)
## !(3.4 in X) is equivalent to !in(3.4, X)
2.5 in X, !(3.4 in X)


# We can overload `one` as follows to create an interval corresponding to $[1,1]$.
# The `one(T)` function will create the "multiplicative identity"
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

# We now want to overload the operations `+`, `/` and `*` so that we can compute the Taylor
# series of `exp`. We begin with `+`. 

function +(X::Interval, Y::Interval)
    a,b,c,d = promote(X.a, X.b, Y.a, Y.b) # make sure all are the same type
    T = typeof(a)
    α = setrounding(T, RoundDown) do
        a + c
    end
    β = setrounding(T, RoundUp) do
        b + d
    end
    Interval(α, β)
end


+(x::Number, Y::Interval) = Interval(x) + Y # Number is a supertype that contains Int, Float64, etc.
+(X::Interval, y::Number) = X + Interval(y)


## following example was the non-associative example but now we have bounds
Interval(1.1) + Interval(1.2) + Interval(1.3)

## note we are actually doing computations on ${\rm fl}^{nearest}(1.1)$, etc.,
## that is, we haven't accounted in the errors from making the constants. 


# We now implement division, checking that our assumptions 
# are satified. Note that `&&` means "and" whilst `||` means "or",
# While `!` changes a `true` to `false` and vice-versa.


function /(X::Interval, n::Int)
    a,b = promote(X.a, X.b)
    T = typeof(a)
    if !(n > 0 && 0 < a ≤ b)
        error("Input doesn't satisfy positivity assumptions")
    end
    α = setrounding(T, RoundDown) do
            a / n
    end
    β = setrounding(T, RoundUp) do
            b / n
    end
    Interval(α, β)
end

Interval(1.0,2.0)/3 # rounds bottom down and top up

# Finally we overload `*` to behave like the operation `⊗`:

function *(X::Interval, Y::Interval)
    a,b,c,d = promote(X.a, X.b, Y.a, Y.b)
    T = typeof(a)
    if !(0 < a ≤ b && 0 < c ≤ d)
        error("Input doesn't satisfy positivity assumptions")
    end
    α = setrounding(T, RoundDown) do
            a * c
    end
    β = setrounding(T, RoundUp) do
            b * d
    end
    Interval(α, β)
end

# Let's also support powers:

function ^(X::Interval, k::Int)
    if k ≤ 0
        error("not supported")
    elseif k == 1
        X
    else
        X * X^(k-1)
    end
end

# We can now compute positive polynomials with interval arithmetic:

X = Interval(1.0)
1 + X + X^2/2 + X^3/6 + X^4/24

# ------


# **Problem 3(a)** Complete the following implementations of `-` to correctly round
# the endpoints in interval negation and subtraction.

import Base: -

function -(X::Interval)
    a,b = promote(X.a, X.b)
    ## TODO: return an interval representing {-x : x in X}
    
end

function -(X::Interval, Y::Interval)
    a,b,c,d = promote(X.a, X.b, Y.a, Y.b)
    T = typeof(a)
    ## TODO: return an interval implementing X ⊖ Y
    
end

@test -Interval(0.1,0.2) == Interval(-0.2, -0.1)
@test Interval(0.1,0.2) - Interval(1.1,1.2) ≡ Interval(-1.1, -0.9)

# **Problem 3(b)** Alter the implementation of `/(X::Interval, n::Int)`
# to support the case where `n < 0` and `*` to remove the restrictions on
# positivity of the endpoints. You may assume the intervals are non-empty.

## TODO: overload / and *, again.



@test Interval(1.1, 1.2) * Interval(2.1, 3.1) ≡ Interval(2.31, 3.72)
@test Interval(-1.2, -1.1) * Interval(2.1, 3.1) ≡ Interval(-3.72, -2.31)
@test Interval(1.1, 1.2) * Interval(-3.1, -2.1) ≡ Interval(-3.72, -2.31)
@test Interval(-1.2, -1.1) * Interval(-3.1, -2.1) ≡ Interval(2.31, 3.72)


@test Interval(1.0,2.0)/3 ≡ Interval(0.3333333333333333, 0.6666666666666667)
@test Interval(1.0,2.0)/(-3) ≡ Interval(-0.6666666666666667, -0.3333333333333333)

@test Interval(-1., 2) * Interval(2,3) ≡ Interval(-3.0, 6.0)
@test Interval(-1., 2) * Interval(-3,5) ≡ Interval(-6.0, 10.0)

# -----

# The following function  computes the first `n+1` terms of the Taylor series of $\exp(x)$:
# $$
# \sum_{k=0}^n {x^k \over k!}
# $$
# We avoid using `factorial` to avoid underflow/overflow.

function exp_t(x, n)
    ret = one(x)
    s = one(x)
    for k = 1:n
        s = s/k * x
        ret = ret + s
    end
    ret
end

exp_t(X, 100) # Taylor series with interval arithemtic


# In the notes we derived a bound assuming $0 ≤ x ≤ 1$
# on the error in Taylor series of the form $|δ_{x,n}| ≤ 3/(n+1)!$.
# Here we incorporate that error to get a rigorous bound.

function exp_bound(X::Interval, n)
    a,b = promote(X.a, X.b)
    T = typeof(a)
    
    if !(0 < a ≤ b)
        error("Interval must be a subset of [0, 1]")
    end
    ret = exp_t(X, n) # the code for Taylor series should work on Interval unmodified
    ## avoid overflow in computing factorial by using `big`.
    ## Convert to type `T` to support rounding.
    f = T(factorial(big(n + 1)),RoundDown)

    δ = setrounding(T, RoundUp) do
        T(3) / f # need to convert 3 to the right type to set the rounding
    end
    ret + Interval(-δ,δ)
end

E = exp_bound(Interval(1.0), 20)

# Here we test that the bounds match our expectations:

@test exp(big(1)) in E
@test E.b - E.a ≤ 1E-13 # we want our bounds to be sharp

# We can even use the code with `BigFloat` to compute a rigorous bound on the first
# 1000 digits of `ℯ`:


e_int_big = setprecision(4_000) do
    exp_bound(Interval(big(1.0)), 1000)
end

# Our tests show that this has computed more than 1000 digits:

@test ℯ in e_int_big # we contain ℯ
@test e_int_big.b - e_int_big.a ≤ big(10.0)^(-1200) # with 1200 digits of accuracy!



# ------
# **Problem 4** Extend the implementation of `exp` for the case when `-2 ≤ x ≤ 2`.

## TODO: re-overload `exp` but without the restrictions on positivity and adjusting the
## the bound appropriately.



@test exp(big(-2)) in exp_bound(Interval(-2.0), 20)

# **Problem 5(a)** Complete the implementation of a function `sin_t(x,n)` computing the
# first `2n+1` terms of the Taylor series:
# $$
# \sin\ x ≈ ∑_{k=0}^n {(-1)^k x^{2k+1} \over (2k+1)!}
# $$

function sin_t(x, n)
    ret = x
    s = x
    ## TODO: Compute the first 2n+1 terms of the Taylor series of sin, without using the factorial function
    
    ret
end

@test sin_t(1.0, 10) ≈ 0.8414709848078965
@test sin_t(big(1.0), 10) in  sin_t(Interval(1.0), 10)

# **Problem 5(b)** Complete the implementation of a function `sin_bound(x,n)` that
# includes an error bound on the computation. You may assume $0 ≤ x ≤ 1$.

function sin_bound(X::Interval, n)
    a,b = promote(X.a, X.b)
    T = typeof(a)
    ## TODO: complete the implementation to include the error in truncating the Taylor series. 
    
end


S = sin_bound(Interval(1.0), 20)
@test sin(big(1)) in S
@test S.b - S.a ≤ 1E-13 # we want our bounds to be sharp