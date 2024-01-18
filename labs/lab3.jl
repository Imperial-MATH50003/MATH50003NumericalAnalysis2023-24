# # MATH50003 (2023–24)
# # Lab 3: II.1 Integers and II.2 Reals

# In this lab, we will explore how a computer represents integers (both signed and unsigned) and reals.
# In particular, its usage of modular and floating point arithmetic.



# **Learning Outcomes**
#
# Mathematical knowledge:
#
# 1. Behaviour of modular arithmetic for signed and unsigned integers.
# 2. Binary and hexadecimal integer formats.
# 3. Overflow behaviour for integers.
#
# Coding knowledge:
#
# 1. Creating numbers with different formats via `UInt8`, `Float16`, etc.
# 2. The `sizeof`, `reinterpret`, `parse`, `typemax`, `bitstring`, and `printbits` functions
# 3. String construction and concatenation via `*`.
# 4. Creating floating point numbers by specifying their bits.


# We load an external `ColorBitstring` package
# which implements functions `printbits` (and `printlnbits`)
# to print the bits (and with a newline) of numbers in colour:

using ColorBitstring, Test

# If this fails you may need to call `] add ColorBitstring`.

# ## II.1 Integers

# We now explore the representation and behaviour of integers in Julia,
# which is identical to other compiled languages like C, Swift, Rust, etc.
# The basic integer type, which is constructed when we write say `5` is
# the signed integer type `Int`:

typeof(5)

# On a 64-bit machine this will return `Int64` indicating it is using 64-bits.
# (It's possible your machine is 32-bit in which case this will be `Int32`.)
# But other integer types exist, in particular for unsigned integers and for
# different number of bits.

# Every primitive number type is stored as a sequence of bits.
# The number of _bytes_ (i.e. 8-bits) can be deduced using the `sizeof` function:

sizeof(Int64) # 8 bytes == 8*8 bits == 64 bits

# We will use the `printbits` command provided by `ColorBitstring` to see what bits
# are actually stored, eg.

printbits(5)



# ### II.1.1 Unsigned integers

# Unsigned integers are used to represent non-negative integers. In Julia
# these correspond to types `UInt8`, `UInt16`, etc. where the number indicates
# the number of bits. The easiest way to create such an integer is to convert
# from an `Int`:

UInt8(5) # creates an Int and converts it to an UInt8
         # displaying the result in hex

# This fails if a number cannot be represented as a specified type:
# e.g. `UInt8(-5)` and `UInt8(2^8)`.

# (These can also be written as e.g. `convert(UInt8, 5)`.)
# We can also create unsigned integers by specifying their bits
# by writing `0b` followed by a sequence of bits:

0b101 # creates an UInt8, the smallest type with at least 3 bits
#
0b10111011101 # creates an UInt16, the smallest type with at least 11 bits


# -----
# **Problem 1(a)** Use binary format to create an `UInt32` corresponding to $(101101)_2$.

## TODO: Create an UInt32 representing (101101)_2



# **Problem 1(b)** What happens if you specify more than 64 bits using `0b⋅⋅…⋅⋅`?
# What if you specify more than 128 bits?

## TODO: Experiment with 0b with different amounts of digits.


# -----



# Integers use modular arithmetic for addition, subtraction and multiplication:

x = UInt8(17)  # An 8-bit representation of the number 255, i.e. with bits 00010001
y = UInt8(3)   # An 8-bit representation of the number   1, i.e. with bits 00000011
printbits(x); println(" + "); printbits(y); println(" = ")
printlnbits(x + y) # + is automatically modular arithmetic
printbits(x); println(" - "); printbits(y); println(" = ")
printbits(x - y) # - is automatically modular arithmetic



# If we go past the largest integer we overflow and wrap around:

x = UInt8(255) # An 8-bit representation of the number 255, i.e. with bits 11111111
y = UInt8(1)   # An 8-bit representation of the number   1, i.e. with bits 00000001
printbits(x); println(" + "); printbits(y); println(" = ")
printbits(x + y) # + is automatically modular arithmetic

# A similar phenomena happens with subtraction:

x = UInt8(3) # An 8-bit representation of the number   3, i.e. with bits 00000011
y = UInt8(5) # An 8-bit representation of the number   5, i.e. with bits 00000101
printbits(x); println(" - "); printbits(y); println(" = ")
printbits(x - y) # + is automatically modular arithmetic


# Multiplication also works similarly. Multiplication by two shifts bits by
# one and modular arithmetic just drops extra bits so we have the following behaviour:

x = UInt8(254) # An 8-bit representation of the number 254, i.e. with bits 11111110
y = UInt8(2)   # An 8-bit representation of the number   2, i.e. with bits 00000010
printbits(x); println(" * "); printbits(y); println(" = ")
printbits(x * y) # represents 252

# ### II.1.2 Signed integers

# Signed integers represent negative and non-negative integers, where
# if the first bit is `1` it is interpreted as a negative number, according to
# the 2's complement format. There are multiple types of signed integers including
# `Int8`, `Int16`, `Int32`, and `Int64`. By default we create an `Int` but we can
# convert an `Int` to another signed integer type:

Int8(5) # display of Int8 does not reveal its type

# It prints the same as `5` but calling `typeof` will confirm it is indeed an `Int8`.
# We can use `printbits` to see the expected binary format, matching that of `UInt8(5)`:

printbits(Int8(5)) # 5 = 2^2 + 1

# Negative numbers use 2's complement, for example we have

printbits(Int8(-5)) # -5 mod 256 = 251 = 1 + 2 + 2^3 + 2^4 + 2^5 + 2^6 + 2^7

# We can use the `reinterpret` function to create an unsigned integer by
# specifying a sequence of bits and reinterpreting the bits as a signed integer:

reinterpret(Int8, 0b11111111) # Create an Int8 with the bits 11111111

# This is different from conversion via `Int8(0b11111111)` (which throws an error):
# `0b11111111` represents the (unsigned) integer $2^8-1 = 255$ and hence
# `Int8(0b11111111)` is equivalent to `Int8(UInt8(255))`. Since `255` is larger than
# the largest `Int8` (which is $2^7-1 = 127$) it would through an error.


# -----

# **Problem 2(a)** Construct an `Int8` representing the number $-63$ by specifying its
# bits and using `reinterpret`.

## TODO: Create an unsigned integer with 8 bits and reinterpret to get Int8(-63)


# **Problem 2(b)** Can you predict what the output of the following will be before hitting return?
# Check that you are correct.

UInt8(120) + UInt8(10) # Convert to Int to see the number printed in decimal
#
Int8(120) + Int8(10)
#
UInt8(2)^7
#
Int8(2)^7
#
Int8(2)^8
#





# ### Strings and parsing

# Strings are a convenient way of representing arbitrary strings of digits:
# we can convert bits of a number to a string of "1"s and "0"s
# using the function `bitstring`. For example:

bitstring(Int8(5))

# Whereas `printbits` prints the bits, this actually returns a string
# that can further be manipulated.

# -----

# **Problem 3(a)** Can you predict what the output of the following will be before hitting return?

bitstring(11)
#
bitstring(-11)



# -----

# We can `parse` a string of digits in base 2 or 10:

parse(Int8, "11"; base=2), # represents 2 + 1 = 3 as an Int8
parse(Int8, "00001011"; base=2) # represents 2^3 + 2 + 1 = 11 as an Int8

# Be careful with "negative" numbers, the following will fail: `parse(Int8, "10001011"; base=2)`

# It treats the string as binary digits, NOT bits. That is, negative numbers
# are represented using the minus sign:

parse(Int8, "-00001011"; base=2)

# -----

# **Problem 3(b)** Combine `parse`, `reinterpret`, and `UInt8` to convert the
# above string to a (negative) `Int8` with the specified bits.

## TODO: combine parse and reinterpret 


# -----

# To concatenate strings we can use `*` (multiplication is used because string concatenation
# is non-commutative):

"hi" * "bye"

# The string consisting of the first nine characters can be found using `str[1:9]` where `str` is any string:

str = "hibye0123445556"
str[1:9]  # returns "hibye0123"

# The string consisting of the 11th through last character can be found using `str[11:end]`:

str = "hibye0123445556"
str[11:end]  # returns "45556"

# We can combine string manipulation with `bitstring` and `parse` to manipulate bits.
# For example, we can see which `Int8` has the same bits as `Int8(5)` but with the third bit
# set to 1.

str = bitstring(Int8(5)) # string of bits for 5, eg  "00000101"
tru = str[4:end] # drop first four characters of the string, eg "000101"
swa = str[1:2] * "1" * tru # add the character "1" at the third position, eg "00100101"
parse(Int8, swa; base=2) # answer is 37 = 5 + 2^5

# -----

# **Problem 3(c)** Complete the following function that sets the 10th bit of an `Int32` to `1`,
# and returns an `Int32`, assuming that the input is a positive integer, using `bitstring`,
# `parse` and `*`.

function tenthbitto1(x::Int32)
    ## TODO: change the 10th bit of x to 1
    
end

@test tenthbitto1(Int32(100)) ≡ Int32(4194404)

# ### II.1.3 Hexadecimal and binary format

# In Julia unsigned integers are displayed in hexadecimal
# form: that is, in base-16.
# Since there are only 10 standard digits (`0-9`) it uses 6 letters (`a–f`) to represent
# 11–16. For example,

UInt8(250)

# because `f` corresponds to 15 and `a` corresponds to 10, and we have
# $$
# 15 * 16 + 10 = 250.
# $$
# The reason for this is that each hex-digit encodes 4 bits (since 4 bits have $2^4 = 16$ possible
# values) and hence two hex-digits are encode 1 byte, and thus the digits correspond
# exactly with how memory is divided into addresses.
# We can create unsigned integers either by specifying their hex format:

0xfa

# Alternatively, we can specify their digits.
# For example, we know $(f)_{16} = 15 = (1111)_2$ and $(a)_{16} = 10 = (1010)_2$ and hence
# $250 = (fa)_{16} = (11111010)_2$ can be written as

0b11111010






# -----

# ## II.2 Reals
#
# Real numbers interpret a sequence of bits as a real number, specified in
# floating point.
# In Julia these correspond to 3 different floating-point types:

# `Float64` is a type representing double precision ($F_{64} = F_{1023,11,52}$).
# We can create a `Float64` by including a
# decimal point when writing the number:

5.3 # isa Float64

# We can use `printbits` to see the stored bits:

printbits(5.125)

# The red bit is the sign bit (0 means positive). The Green bits represent the exponent, in this case:

σ = 1023 # the shift according to Float64 format
0b10000000001 - σ # == 2

# The blue bits are the significand. In this case represent `(1.01001)_2 = 1 + 2^(-2) + 2^(-5)`. And indeed
# we have
# $$
# 2^2 (1+2^{-2} + 2^{-5}) = 5 + 2^{-3} = 5.125
# $$

# Alternatively, one can use scientific notation: `5.125e0` to construct a `Float64`.
# `Float64` is the default format for
# scientific computing.
#
# `Float32` is a type representing single precision ($F_{32} = F_{127,8,23}$).  We can create a `Float32` by including a
# `f0` when writing the number. Here we create a `Float32` and print its bits:

printbits(5.125f0) # 5.125f0 of type Float32

# Now the exponent is

σ = 127 # the shift according to Float32 format
0b10000001 - σ # == 2

# and again we see this represents `5.125`.
# `Float32` is generally the default format for graphics (on the _Graphics Processing Unit_, GPU),
# as the difference between 32 bits and 64 bits is indistinguishable to the eye in visualisation,
# and more data can be fit into a GPU's limited memory.

# `Float16` is a type representing half-precision ($F_{16} = F_{15,5,10}$).
# It is important in machine learning where one wants to maximise the amount of data
# and high accuracy is not necessarily helpful.  We create `Float16` by converting a `Float64`
# as follows:

printbits(Float16(5.125))

# Now the exponent is

σ = 15 # the shift according to Float16 format
0b10001 - σ # == 2

# and we are representing $5.125$ again.
# `Float16` is important in machine learning where one wants to maximise the amount of data
# and high accuracy is not necessarily helpful.
#
# -----

# **Problem 4(a)** Use `reinterpret` and binary format to deduce which `Float16` has bits
# `1 01010 1010101010`.

## TODO: Construct a Float16 with the bits 1 01010 1010101010



#

# **Problem 4(b)** Use `printbits` to guess the binary representation of $1/5$.

## TODO: Use printbits and guess the binary expansion of 1/5.


# -----


# We now construct the largest and smallest `Float32` and check their bit sequences:

σ,Q,S = 127,8,23 # Float32
εₘ = 2.0^(-S)
printlnbits(Float32(2.0^(1-σ))) # smallest positive normal Float32
printlnbits(Float32(2.0^(2^Q-2-σ) * (2-εₘ))) # largest normal Float32

# For a given floating-point type, we can find these constants using the following functions:

eps(Float32), floatmin(Float32), floatmax(Float32)

# ### II.2.4 Sub-normal and special numbers

# If all the exponent bits are `0` then the number represents a "sub-normal" floating point number.

# **Example (creating a sub-normal number)** If we divide the smallest normal number by `2`, we get a subnormal number:

mn = floatmin(Float32) # smallest normal Float32
printlnbits(mn)
printbits(mn/2)

# Can you explain the bits?
#
# Zero is a sub-normal number, but it turns out there is also a negative zero:

printlnbits(0.0) # 0 has all bits 0
printlnbits(-0.0) # -0 has sign bit 1 and all other bits zero

#
# -----


# **Problem 4(c)** Create the smallest positive non-zero sub-normal `Float16` by specifying
# its bits.

## TODO: create the smallest positive Float16


# -----

# The special numbers extend the real line by adding $±∞$ but also a notion of "not-a-number" ${\rm NaN}$.
# Whenever the bits of the exponent $q$ of a floating-point number are all 1 then they represent an element of $F^{\rm special}$.
# If all $b_k=0$, then the number represents either $±∞$, called `Inf` and `-Inf` for 64-bit floating-point numbers (or `Inf16`, `Inf32`
# for 16-bit and 32-bit, respectively):

printlnbits(Inf16)
printbits(-Inf16)

# All other special floating-point numbers represent ${\rm NaN}$. One particular representation of ${\rm NaN}$
# is denoted by `NaN` for 64-bit floating-point numbers (or `NaN16`, `NaN32` for 16-bit and 32-bit, respectively):

printbits(NaN16)


# Arithmetic works differently on `Inf` and `NaN` and for undefined operations. 
# In particular we have:

1/0.0        # returns  Inf
1/(-0.0)     # returns -Inf
0.0/0.0      # returns  NaN
  
Inf*0        # returns  NaN
Inf+5        # returns  Inf
(-1)*Inf     # returns -Inf
1/Inf        # returns  0.0
1/(-Inf)     # returns -0.0
Inf - Inf    # returns  NaN
Inf ==  Inf  # returns  true
Inf == -Inf  # returns  false

NaN*0        # returns  NaN
NaN+5        # returns  NaN
1/NaN        # returns  NaN
NaN == NaN   # returns  false
NaN != NaN   # returns  true


# Essentially `NaN` is a CPU's way of indicating an error has occurred, but computation
# will continue.




# **Example (many `NaN`s)** What happens if we change some other $b_k$ to be nonzero?
# We can create bits as a string and see:

i = 0b0111110000010001 # an UInt16
reinterpret(Float16, i)

# Thus, there are many ways of representing `NaN`. (What a waste of perfectly good bit sequences!)