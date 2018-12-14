# TRS.jl: Solving the Trust Region Subproblem as an eigenproblem in Julia

This package solves the Trust-Region subproblem:
```
minimize    ½x'Px + q'x
subject to  ‖x‖ = r
```
where `x` in the `n-`dimensional variable, in pure Julia. This is a `matrix-free` method accessing `P` *only* via matrix multiplications (i.e. via `mul!`), so it can take full advantage of `P`'s structure/sparsity.

This package implements the algorithm described in [Adachi et al 2017](https://epubs.siam.org/doi/abs/10.1137/16M1058200). Although an implementation can be written in just three lines this package provides a carefully written, efficient implementation.

The following extensions are supported:
* Ellipsoidal norms: `‖x‖ = sqrt(x'Cx)` for any positive definite C.
* Linear equality constraints: `Ax = b`
* Degenerate cases (e.g. the so called `hard case`)
* Finding the local-no-global minimizer.

# Function reference:
## Vanilla TRS
```
x = trs(P, q, r; options)
```
where options 

## 

# Finding local-no-global minimizers.
```
x = trs(P, q, r; compute_local=true)
x = trs(P, q, r; x0, compute_local=true)
```

Solving the TRS is a matter of only three lines (using `Arpack.jl`):
```
λ, v = eigs!([-P I; q*q'/r^2 -P], nev=1, which:LR)
v1 = v[1:n]; v2 = v[n+1:end]
x = -v1*sign(q'*v2)*r/norm(v1)
```
