# MATH50003NumericalAnalysis
Notes and course material for MATH50003 Numerical Analysis (2023–24)

Lecturer: [Dr Sheehan Olver](https://www.ma.imperial.ac.uk/~solver/)

Office hour: Mondays 11am, Huxley 6M40

## [Notes](https://github.com/Imperial-MATH50003/MATH50003NumericalAnalysis/raw/main/notes/notes.pdf) (up to Week 4)

## Weekly Material

1. I.1 Rectangular Rule and I.2 Divided Differences
   - [Lab 1](https://github.com/Imperial-MATH50003/MATH50003NumericalAnalysis/blob/main/labs/lab1.ipynb) ([Solutions](https://github.com/Imperial-MATH50003/MATH50003NumericalAnalysis/blob/main/labs/lab1s.ipynb))
   - [Sheet 1](https://github.com/Imperial-MATH50003/MATH50003NumericalAnalysis/blob/main/sheets/sheet1.pdf) ([Solutions](https://github.com/Imperial-MATH50003/MATH50003NumericalAnalysis/blob/main/sheets/sheet1s.pdf))
2. I.3 Dual Numbers and I.4 Newton's method
   - [Lab 2](https://github.com/Imperial-MATH50003/MATH50003NumericalAnalysis/blob/main/labs/lab2.ipynb) ([Solutions](https://github.com/Imperial-MATH50003/MATH50003NumericalAnalysis/blob/main/labs/lab2s.ipynb))
   - [Sheet 2](https://github.com/Imperial-MATH50003/MATH50003NumericalAnalysis/blob/main/sheets/sheet2.pdf) ([Solutions](https://github.com/Imperial-MATH50003/MATH50003NumericalAnalysis/blob/main/sheets/sheet2s.pdf))
3. II.1 Integers and II.2 Reals
   - [Lab 3](https://github.com/Imperial-MATH50003/MATH50003NumericalAnalysis/blob/main/labs/lab3.ipynb) ([Solutions](https://github.com/Imperial-MATH50003/MATH50003NumericalAnalysis/blob/main/labs/lab3s.ipynb))
   - [Sheet 3](https://github.com/Imperial-MATH50003/MATH50003NumericalAnalysis/blob/main/sheets/sheet3.pdf) ([Solutions](https://github.com/Imperial-MATH50003/MATH50003NumericalAnalysis/blob/main/sheets/sheet3s.pdf))
4. II.3 Floating Point Arithmetic and II.4 Interval Arithmetic
   - [Lab 4](https://github.com/Imperial-MATH50003/MATH50003NumericalAnalysis/blob/main/labs/lab4.ipynb)
   - [Sheet 4](https://github.com/Imperial-MATH50003/MATH50003NumericalAnalysis/blob/main/sheets/sheet4.pdf)



## Assessment

1. Practice computer-based exam (marked but not for credit): TBA
2. Computer-based exam: 15 March 2024, 9:00–10:30
5. Final exam (pen-and-paper): TBA


## Reading List

1. [MATH50003 Introduction to Julia](https://github.com/Imperial-MATH50003/MATH50003NumericalAnalysis/blob/main/notes/A.Julia.ipynb)
4. [The Julia Documentation](https://docs.julialang.org)
6. Ben Lauwens, [Think Julia](https://benlauwens.github.io/ThinkJulia.jl/latest/book)
5. [The Julia–Matlab–Python Cheatsheet](https://cheatsheets.quantecon.org)
2. Tobin A. Driscoll & Richard J. Braun, [Fundamentals of Numerical Computation, Julia Edition](https://tobydriscoll.net/fnc-julia/linsys/overview.html), Chapters 1–3, 7
2. Nicholas J. Higham, [Accuracy and Stability of Numerical Algorithms](https://epubs.siam.org/doi/book/10.1137/1.9780898718027?mobileUi=0), Chapters 1–3
1. Michael L. Overton, [Numerical Computing with IEEE Floating Point Arithmetic](https://epubs.siam.org/doi/book/10.1137/1.9780898718072), Chapters 2–6
2. Lloyd N. Trefethen & David Bau III, [Numerical Linear Algebra](https://my.siam.org/Store/Product/viewproduct/?ProductId=950/&ct=c257a1956367c57b599612fbf383d0d3c674af4f9181d827444b5cdaca95b0686d6d20467a7c1e3290fb5b31c310ce74f5b2ede375934b844b1171bc734358e2), Chapters 1–4
3. Lloyd N. Trefethen, [Approximation Theory and Approximation Practice](https://people.maths.ox.ac.uk/trefethen/ATAP/ATAPfirst6chapters.pdf), Chapters 1–4, 17–19
7. David A. Ham, [Just enough Git to get by](https://object-oriented-python.github.io/a2_git.html)


## What is numerical analysis?

Broadly speaking, numerical analysis is the study of approximating
solutions to _continuous problems_ in mathematics, for example, integration, differentiation,
and solving differential equations. There are three key topics in numerical analysis:

1. Design of algorithms: discuss the construction of algorithms and their implmentation in
software.
2. Convergence of algorithms: proving under which conditions algorithms converge to the
true solution, and the rate of convergence.
3. Stability of algorithms: the study of robustness of algorithms to small perturbations in
data, for example, those that arise from measurement error, errors if data are themselves computed using
algorithms, or simply errors arising from the way computers represent real numbers.

The modern world is built on numerical algorithms:


1. [Fast Fourier Transform (FFT)](https://en.wikipedia.org/wiki/Fast_Fourier_transform): Gives a highly efficient way of computing Fourier  coefficients from function samples,
and is used in many places, e.g., the mp3 format for compressing audio and JPEG image format.
(Though, in a bizarre twist, GIF, a completely uncompressed format, has made a remarkable comeback.)
2. [Singular Value Decomposition (SVD)](https://en.wikipedia.org/wiki/Singular_value_decomposition): Allows for approximating matrices by those with low rank. This is related to the [PageRank algorithm](https://en.wikipedia.org/wiki/PageRank) underlying Google.
3. [Stochastic Gradient Descent (SGD)](https://en.wikipedia.org/wiki/Stochastic_gradient_descent): Minima of high-dimensional functions can be effectively computed using gradients
in a randomised algorithm. This is used in the training of machine learning algorithms.
4. [Finite element method (FEM)](https://en.wikipedia.org/wiki/Finite_element_method):
used to solve many partial differential equations including  in aerodynamics and
weather prediction. [Firedrake](https://firedrakeproject.org) is a major project based at
Imperial that utilises finite element method.


This is not to say that numerical analysis is only important in applied mathematics.
It is playing an increasing important role in pure mathematics with important proofs based on numerical computations:

1. [The Kepler conjecture](https://en.wikipedia.org/wiki/Kepler_conjecture). This 400 year conjecture on optimal sphere packing
was finally proved in 2005 by Thomas Hales using numerical linear programming.
2. [Numerical verification of the Riemann Hypothesis](https://en.wikipedia.org/wiki/Riemann_hypothesis#Numerical_calculations).
It has been proved that there are no zeros of the Riemann zeta function off the critical line provided the imaginary part is
less than 30,610,046,000.
3. [Steve Smale's 14th problem](https://en.wikipedia.org/wiki/Lorenz_system) on the stability of the Lorenz system was solved
using interval arithmetic.

Note these proofs are _rigorous_: as we shall see it is possible to obtain precise error bounds in numerical
calculations, and the computations themselves can be computer-verified
(a la [The Lean Theorem Prover](https://leanprover.github.io)).
As computer-verified proofs become increasingly important, the role of numerical analysis in
pure mathematics will also increase, as it provides the theory for rigorously controlling errors in
computations. Note that there are many computer-assisted proofs that do not fall under numerical analysis because
they do not involve errors in computations or approximation or involve discrete problems, for
example, the proof the [Four Colour Theorem](https://en.wikipedia.org/wiki/Four_color_theorem).

## The Julia Programming Language

In this course we will use the programming language [Julia](https://julialang.org). This is a modern, compiled, high-level,
open-source language developed at MIT. It is becoming increasingly important in high-performance computing and
AI, including by Astrazeneca, Moderna and Pfizer in drug development and clinical trial accelleration, IBM for medical diagnosis, MIT for robot
locomotion, and elsewhere.

It is ideal for a course on numerical analysis because it both allows
_fast_ implementation of algorithms but also has support for _fast_ automatic-differentiation, a feature
that is of increasing importance in machine learning. It also is low level enough that we can
really understand what the computer is doing. As a bonus, it is easy-to-read and fun to write.

To run Julia in a Jupyter notebook on your own machine:

1. Download [Julia v1.10](https://julialang.org/downloads/)
2. Open the Julia app which will launch a new window
3. Install the needed packages by typing (`]` will change the prompt to a package manager):
```julia
] add IJulia Plots ColorBitstring SetRounding
```
3. Build Jupyter via
```julia
] build IJulia
```
4. Launch Jupyter by typing
```julia
using IJulia
notebook()
```
5. Download the labs to the same folder as the Jupyter notebook is running.

## Past exams


1. [2021–22 Computer-based exam](https://github.com/Imperial-MATH50003/MATH50003NumericalAnalysis/blob/main/exams/computerexam2122.ipynb)
2. [2022–23 Computer-based exam](https://github.com/Imperial-MATH50003/MATH50003NumericalAnalysis/blob/main/exams/computerexam2223.ipynb)

## Past Course Websites

1. [2021-22](https://github.com/Imperial-MATH50003/MATH50003NumericalAnalysis2021-22)
2. [2022-23](https://github.com/Imperial-MATH50003/MATH50003NumericalAnalysis2022-23)
