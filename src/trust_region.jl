mutable struct TRSinfo
	status::Symbol
	niter::Int # Number of iterations in eigs
	nmul::Int  # Number of multiplication with P in eigs
	λ  # Lagrange Multiplier(s)

	function TRSinfo(status::Symbol, niter::Int, nmul::Int, λ)
		new(status, niter, nmul, λ)
	end
end

function trs(P, q::AbstractVector{T}, r::T, tol::T=1e-13) where {T}
	x1, info = trs_boundary(P, q, r, tol)
	if info.λ <= 0 # Global solution is in the interior
		cg!(x1, P, -q, 1e-12)
		info.λ = 0
	end
	return x1, info
end

function trs(P, q::AbstractVector{T}, r::T, compute_local, tol::T=1e-13) where {T}
	x1, x2, info = trs_boundary(P, q, r, tol)
	if info.λ[1] <= 0 # Global solution is in the interior
		cg!(x1, P, -q, 1e-12)
		info.λ[1] = 0
	end
	if info.λ[2] <= 0
		# No local-no-global minimiser can exist in the interior
		x2 = []
		info.λ[2] = NaN
	end
	return x1, x2, info
end

function trs_boundary(P, q::AbstractVector{T}, r::T, tol::T=1e-13) where {T}
	check_inputs(P, q, r)
	return trs_boundary(() -> eigenproblem(P, q, r, 1, tol),
		   (λ, V) -> pop_solution!(P, q, r, V, λ))
end

function trs_boundary(P, q::AbstractVector{T}, r::T, compute_local, tol::T=1e-13) where {T}
	check_inputs(P, q, r)
	return trs_boundary(() -> eigenproblem(P, q, r, 2, tol),
		   (λ, V) -> pop_solution!(P, q, r, V, λ),
		true)
end

function trs_boundary(solve_eigenproblem::Function, pop_solution!::Function)
	λ, V, niter, nmult = solve_eigenproblem()
	x1, x2, λg = pop_solution!(λ, V)
	if !isempty(x2)  # We can only have two global solutions in the "hard-case"
		status = :G
	else
		status = :GH
	end
	return x1, TRSinfo(status, niter, nmult, λg)
end

function trs_boundary(solve_eigenproblem::Function, pop_solution!::Function, compute_local)
	λ, V, niter, nmult = solve_eigenproblem()
	x1, x2, λg = pop_solution!(λ, V) # Pop global minimizer(s).
	if isempty(x2) # i.e. we are not in the hard-case
		x2, _, λl = pop_solution!(λ, V) # Pop local-no-global minimizer.
		if !isempty(x2)
			status = :GL # The "local-no-global" minimizer does not exists.
		else
			status = :G # The "local-no-global" minimizer exists.
		end
	else
		status = :GH  # Hard case: thus no "local-no-global" minimizer exist.
	end

	return x1, x2, TRSinfo(status, niter, nmult, [λg λl])
end

function eigenproblem(P, q::AbstractVector{T}, r::T, nev=1, tol::T=1e-13) where {T}
	n = length(q)
	function A(y::AbstractVector, x::AbstractVector)
		@inbounds y1 = view(y, 1:n); @inbounds y2 = view(y, n+1:2*n)
		@inbounds x1 = view(x, 1:n); @inbounds x2 = view(x, n+1:2*n)
		mul!(y1, P, x1)
		mul!(y2, P, x2)
		axpy!(-dot(q, x2)/r^2, q, y1)
		axpy!(-one(T), x1, y2)
	end
	D = LinearMap{T}(A, 2*n; ismutating=true)

	(λ, V, nconv, niter, nmult, resid) = eigs(-D, nev=nev, which=:LR, tol=tol)
	return λ, V, niter, nmult
end

function pop_solution!(P, q::AbstractVector{T}, r::T, V::Matrix{Complex{T}}, λ::Vector{Complex{T}}) where {T}
	n = length(q)

	idx = argmax(real(λ))
	if angle(λ[idx]) >= 1e-6  # No more solutions...
		return zeros(T, 0), zeros(T, 0), NaN
	end
	l = real(λ[idx]);
	λ[idx] = -Inf  # This ensures that the next pop_solution! would not get the same solution.
	v = real(view(V, :, idx)) + imag(view(V, :, idx))
	v1 = view(v, 1:n); v2 = view(v, n+1:2*n)

	tol_hard = 1e-4
	if norm(v1) >= tol_hard
		x1 = -sign(q'*v2)*r*v1/norm(v1)
		x2 = zeros(0)
	else
		y, residual = extract_solution_hard_case(P, q, l, reshape(v1/norm(v1), n, 1))
		nullspace_dim = 3
		while residual >= tol_hard*norm(q) && nullspace_dim <= 9
			κ, W, _ = eigs(P, nev=nullspace_dim, which=:SR)
			y, residual = extract_solution_hard_case!(P, q, r, W[:, abs.(κ + l) .< 1e-6])
			nullspace_dim += 3
		end
		α = roots(Poly([y'*y - r^2, 2*v2'*y, v2'*v2]))
		x1 = y + α[1]*v2
		x2 = y + α[2]*v2
	end

	return x1, x2, l
end

function extract_solution_hard_case(P, q, λ, W)
	D = LinearMap{T}((x) -> W*(W'*(P*x + λ*x)), n; issymmetric=true)
	y = cg(-D, q, tol=1e-12, maxiter=500)
	return y, norm(P*y + λ*y + q)
end


function check_inputs(P, q::AbstractVector{T}, r::T) where {T}
	@assert(issymmetric(P), "The cost matrix must be symmetric.")
	@assert(eltype(P) == T, "Inconsistent element types.")
	@assert(size(P, 1) == size(P, 2) == length(q), "Inconsistent matrix dimensions.")
end