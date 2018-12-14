# TRS.jl: Solving the Trust Region Subproblem as an eigenproblem

This package solves the Trust-Region subproblem:
```
minimize    ½x'Px + q'x
subject to  ‖x‖ ≤ r
```
where `x` in the `n-`dimensional variable. This is a **matrix-free** method obtaining highly accurate solutions efficiently by solving a **single** eigenproblem. It accesses `P` *only* via matrix multiplications (i.e. via `mul!`), so it can take full advantage of `P`'s structure/sparsity.

Furthermore, the following extensions are supported:
* Ellipsoidal norms: `‖x‖ = sqrt(x'Cx)` for any positive definite `C`.
* Linear equality constraints: `Ax = b`
* Degenerate cases (i.e. the so-called `hard case`)
* Finding the local-no-global minimizer.
* Solving the related constant-norm (`‖x‖ = r`) problem for all of the cases described above.

If you are interested for support of linear inequality constraints `Ax ≤ b` check [this](https://no-link-yet.com) package.

The main reference for this package is
```
Adachi, S., Iwata, S., Nakatsukasa, Y., & Takeda, A.
Solving the trust-region subproblem by a generalized eigenvalue problem.
SIAM Journal on Optimization 27.1 (2017): 269-291.
```
Additionally, the cases of local-no-global minimizers and linear equality constraints are covered in
```
Rontsis N., Goulart P.J., & Nakatsukasa, Y.
Solving the extended trust-region subproblem via an active set method.
Preprint in Arxiv.
```

## Documentation
### Standard TRS
The global solution of the standard TRS
```
minimize    ½x'Px + q'x
subject to  ‖x‖ = r,
```
where ‖·‖ is the 2-norm, can be obtained with:
```
trs(P, q, r) -> x, info
```
**Arguments** (`T` is any real numerical type):
* `P`: The quadratic cost represented as any linear operator implementing `mul!`, `issymmetric` and `size`.
* `q::AbstractVector{T}`: the linear cost.
* `r::T`: the radius.

**Output**
* `x::Vector{T}`: The global solution to the TRS
* `info::TRSInfo{T}`: Info structure. See below for details.

### Ellipsoidal Norms
Results for ellipsoidal norms `‖x‖ := sqrt(x'Cx)` can be obtained with
```
trs(P, q, r, C) -> x, info
```
which is the same as `trs(P, q, r)` except for the input argument
* `C::AbstractMatrix{T}`: a positive definite, symmetric, matrix that defines the ellipsoidal norm `‖x‖ := sqrt(x'Cx)`.

**N.B.**: `trs(P, q, r, C)` solves a considerably different eigenproblem (a generalized eigenvalue problem as compared to the unsymmetric standard eigenproblem of `trs(P, q, r)`). The user might prefer to perform a change of variables `y = cholesky(C)\x` and use `trs(P, q, r)` in cases where:
* Repeated problems need to be solved with the same `C`; and/or
* A high accuracy solutions is not required.

### Equality constraints
The problem
```
minimize    ½x'Px + q'x
subject to  ‖x‖ ≤ r
            Ax = b
```
can be solved as
```
trs(P, q, r, F) -> x, info
```
which is the same as `trs(P, q, r)` for the input argument `F` which can either be
* `F::Factorization{T}`: a factorization of the matrix `[I A'; A 0]`; or
* `F::function`: a mutating function `F(y, z)` which writes into `y` the projection of `z` into the nullspace of `A`.
### Finding local-no-global minimizers.
Due to non-convexity, TRS can exhibit at most one local minimizer with objective value less than the one of the global. This can be obtained via:
```
trs(P, q, r, [F, C::AbstractMatrix{T}=I}], compute_local=true) -> x1, x2, info
```
* If we do not belong the *hard-case* (i.e. almost always):  
`x1::Vector{T}` is the global solution; and  
`x2::Vector{T}` is the local-no-global one.
* If we do belong to the *hard-case*:  
`x1::Vector{T}` and `x2::Vector{T}` correspond to distinct global minimizers.  
(The reader might recall that in the hard case the local-no-global minimizer does not exist but there exist at least distinct two global minimizers.)

The user can detect the hard case via the returned symbol in `info.status`.

### Solving constant norms
Simply use `trs_boundary` instead of `trs`.

### The `TRSInfo` structure
The returned info structure contains the following 
* `status::Symbol`
* `niter::Int`:  Number of iterations of the eigensolver
* `nmul::Int`:   Number of multiplications with P requested by the eigensolver.
* `λ::Vector{T}` Lagrange Multiplier(s) of the solution(s).