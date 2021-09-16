# TRS.jl: Solving the Trust Region Subproblem

This package solves the Trust-Region Subproblem:
```
minimize    ½x'Px + q'x
subject to  ‖x‖ ≤ r
```
where `x` in the `n-`dimensional variable. This is a **matrix-free** method returning highly accurate solutions efficiently by solving a **single** eigenproblem. It accesses `P` *only* via matrix multiplications (i.e. via `mul!`), so it can take full advantage of `P`'s structure/sparsity.

Furthermore, the following extensions are supported:
- [TRS.jl: Solving the Trust Region Subproblem](#trsjl-solving-the-trust-region-subproblem)
  - [Installation](#installation)
  - [Documentation](#documentation)
    - [Standard TRS](#standard-trs)
    - [Ellipsoidal Norms](#ellipsoidal-norms)
    - [Equality constraints](#equality-constraints)
    - [Finding local-no-global minimizers](#finding-local-no-global-minimizers)
    - [Solving constant-norm problems](#solving-constant-norm-problems)
    - [Solving small problems](#solving-small-problems)
    - [The `TRSInfo` struct](#the-trsinfo-struct)

This package has been specifically designed for large scale problems. Separate, efficient [functions for small problems](#solving-small-problems) are also provided.

If you are interested for support of linear inequality constraints `Ax ≤ b` check [this](https://github.com/oxfordcontrol/QPnorm.jl) package.

The main references for this package are
```
Rontsis N., Goulart P.J., & Nakatsukasa, Y.
An active-set algorithm for norm constrained quadratic problems
Mathematical Programming (2021): 1-37.
```
and
```
Adachi, S., Iwata, S., Nakatsukasa, Y., & Takeda, A.
Solving the trust-region subproblem by a generalized eigenvalue problem.
SIAM Journal on Optimization 27.1 (2017): 269-291.
```

## Installation
This package can be installed by running
```
add https://github.com/oxfordcontrol/TRS.jl
```
in [Julia's Pkg REPL mode](https://docs.julialang.org/en/v1/stdlib/Pkg/index.html#Getting-Started-1).


Alternatively, this package can be installed by running
```
using Pkg
Pkg.add(url="https://github.com/oxfordcontrol/TRS.jl")
```
in a notebook enviroment. 

## Documentation
### Standard TRS
The global solution of the standard TRS
```
minimize    ½x'Px + q'x
subject to  ‖x‖ ≤ r,
```
where ‖·‖ is the 2-norm, can be obtained with:
```julia
trs(P, q, r; kwargs...) -> x, info
```
**Arguments** (`T` is any real numerical type):
* `P`: The quadratic cost represented as any linear operator implementing `mul!`, `issymmetric` and `size`.
* `q::AbstractVector{T}`: the linear cost.
* `r::T`: the radius.

**Output**
* `X::Matrix{T}`: Array with each column containing a global solution to the TRS
* `info::TRSInfo{T}`: Info structure. See [below](#the-trsinfo-struct) for details.

**Keywords (optional)**
* `tol`, `maxiter`, `ncv` and `v0` that are passed to `eigs` used to solve the underlying eigenproblem. Refer to `Arpack.jl`'s [documentation](https://julialinearalgebra.github.io/Arpack.jl/stable/) for these arguments. Of particular importance is **`tol::T`** which essentially controls the **accuracy** of the returned solutions.
* `tol_hard=2e-7`: Threshold for switching to the hard-case. Refer to [Adachi et al.](https://epubs.siam.org/doi/pdf/10.1137/16M1058200), Section 4.2 for an explanation.
* `compute_local::Bool=False`: Whether the local-no-global solution should be calculated. More details [below](#finding-local-no-global-minimizers).

**Note that if `v0` is not set, then `Arpack` starts from a random initial vector and thus the results will not be completely deterministic.**

### Ellipsoidal Norms
Results for ellipsoidal norms `‖x‖ := sqrt(x'Cx)` can be obtained with
```julia
trs(P, q, r, C; kwargs...) -> x, info
```
which is the same as `trs(P, q, r)` except for the input argument
* `C::AbstractMatrix{T}`: a positive definite, symmetric, matrix that defines the ellipsoidal norm `‖x‖ := sqrt(x'Cx)`.

Note that if `C` is known to be well conditioned it might be preferable to perform a change of variables `y = cholesky(C)\x` and use the standard `trs(P, q, r)` instead.

### Equality constraints
The problem
```
minimize    ½x'Px + q'x
subject to  ‖x‖ ≤ r
            Ax = b,
```
where `A` is a "fat", full row-rank matrix, can be solved as
```julia
trs(P, q, r, A, b; kwargs...) -> x, info
```
which is the same as `trs(P, q, r)` except for the input arguments `A::AbstractMatrix{T}` and `b::AbstractVector{T}`

### Finding local-no-global minimizers
Due to non-convexity, a TRS can exhibit at most one local minimizer with objective value less than the one of the global. The local-no-global minimizer can be obtained (if it exists) via:
```julia
trs(···; compute_local=true, kwargs...) -> X info
```
Similarly to the cases above, `X::Matrix{T}` contains the global solution(s), but in this case, local minimizers are also included. The global minimizers(s) proceed the local one.

### Solving constant-norm problems
Simply use `trs_boundary` instead of `trs`.

### Solving small problems
Small problems (say for `n < 20`) should be solved with `trs_small` and `trs_boundary_small`, which have identical definitions with `trs` and `trs_boundary` described above, except for `P` which is constrained to be a subtype of `AbstractMatrix{T}`.

Internally `trs_small`/`trs_boundary_small` use direct eigensolvers (i.e. `eigen`) providing better accuracy, reliability, and speed for small problems.

### The `TRSInfo` struct
The returned info structure contains the following fields:
* `hard_case::Bool` Flag indicating if the problem was detected to be in the hard-case.
* `niter::Int`:  Number of iterations of the eigensolver
* `nmul::Int`:   Number of multiplications with `P` requested by the eigensolver.
* `λ::Vector` Lagrange Multiplier(s) of the solution(s).



