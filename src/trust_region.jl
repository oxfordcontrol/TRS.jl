mutable struct TRSinfo
	is_hard::Bool  # Flag indicating if we are in the hard case
	niter::Int # Number of iterations in eigs
	nmul::Int  # Number of multiplication with P in eigs
	λ::Vector  # Lagrange Multiplier(s)

	function TRSinfo(is_hard::Bool, niter::Int, nmul::Int, λ::Vector)
		new(is_hard, niter, nmul, λ)
	end
end

function trs(P, q::AbstractVector{T}, r::T; compute_local=false, kwargs...) where {T}
	if !compute_local
		x1, info = trs_boundary(P, q, r; compute_local=compute_local, kwargs...)
		if info.λ[1] <= 0 # Global solution is in the interior
			cg!(x1, P, -q)
			info.λ[1] = 0
		end
		return x1, info
	else
		x1, x2, info = trs_boundary(P, q, r; compute_local=compute_local, kwargs...)
		if info.λ[1] <= 0 # Global solution is in the interior
			cg!(x1, P, -q)
			info.λ[1] = 0
		end
		if info.λ[2] <= 0 # Global solution is in the interior
			# No local-no-global minimiser can exist in the interior
			x2 = []
			info.λ[2] = NaN
		end
		return x1, x2, info
	end
end

function trs_boundary(P, q::AbstractVector{T}, r::T; compute_local=false, kwargs...) where {T}
	check_inputs(P, q, r)
	if compute_local
		nev=2
	else
		nev=1
	end
	return trs_boundary((; kw...) -> eigenproblem(P, q, r, nev; kw...),
		   (λ, V; kw...) -> pop_solution!(P, q, r, V, λ; kw...); compute_local=compute_local, kwargs...)
end

function trs_boundary(solve_eigenproblem::Function, pop_solution!::Function;
	compute_local=false, tol_hard=1e-4, kwargs...)
	λ, V, niter, nmult = solve_eigenproblem(kwargs...)
	x1, x2, λ1 = pop_solution!(λ, V; tol_hard=tol_hard) # Pop global minimizer(s).
	if !compute_local
		return x1, TRSinfo(isempty(x2), niter, nmult, [λ1])
	else
		if isempty(x2) # i.e. we are not in the hard-case
			hard_case = false
			x2, _, λ2 = pop_solution!(λ, V) # Pop local-no-global minimizer.
		else
			λ2 = λ1
			hard_case = true
		end
		return x1, x2, TRSinfo(hard_case, niter, nmult, [λ1; λ2])
	end
end

function eigenproblem(P, q::AbstractVector{T}, r::T, nev=1;
				tol=0.0, maxiter=300, v0=zeros((0,))) where {T}
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

	(λ, V, nconv, niter, nmult, _) = eigs(-D, nev=nev, which=:LR,
			tol=tol, maxiter=maxiter, v0=v0)
	@assert(nconv >= nev, "Eigensolver has failed to converge.")

	return λ, V, niter, 2*nmult
end

function pop_solution!(P, q::AbstractVector{T}, r::T, V::Matrix{Complex{T}}, λ::Vector{Complex{T}};
	tol_hard=1e-4) where {T}
	n = length(q)

	idx = argmax(real(λ))
	if abs(real(λ[idx])) <= 1e6*abs(imag(λ[idx])) # No more solutions...
		return zeros(T, 0), zeros(T, 0), NaN
	end
	l = real(λ[idx]);
	λ[idx] = -Inf  # This ensures that the next pop_solution! would not get the same solution.
	v = real(view(V, :, idx)) + imag(view(V, :, idx))
	v1 = view(v, 1:n); v2 = view(v, n+1:2*n)

	if norm(v1) >= tol_hard
		x1 = -sign(q'*v2)*r*v1/norm(v1)
		x2 = zeros(0)
	else
		y, residual = extract_solution_hard_case(P, q, l, reshape(v1/norm(v1), n, 1))
		nullspace_dim = 3
		while residual >= tol_hard*norm(q) && nullspace_dim <= 20 
			κ, W, _ = eigs(P, nev=nullspace_dim, which=:SR)
			y, residual = extract_solution_hard_case(P, q, r, W[:, abs.(κ .+ l) .< 1e-6])
			nullspace_dim *= 2
		end
		α = roots(Poly([y'*y - r^2, 2*v2'*y, v2'*v2]))
		x1 = y + α[1]*v2
		x2 = y + α[2]*v2
	end

	return x1, x2, l
end

function extract_solution_hard_case(P, q::AbstractVector{T}, λ::T, W::AbstractMatrix{T}) where {T}
	D = LinearMap{T}((x) -> P*x + λ*(x + W*(W'*x)), length(q); issymmetric=true)
	y = cg(-D, q)
	return y, norm(P*y + λ*y + q)
end


function check_inputs(P, q::AbstractVector{T}, r::T) where {T}
	@assert(issymmetric(P), "The cost matrix must be symmetric.")
	@assert(eltype(P) == T, "Inconsistent element types.")
	@assert(size(P, 1) == size(P, 2) == length(q), "Inconsistent matrix dimensions.")
end

function gen_eigenproblem(P, q::AbstractVector{T}, r::T, C::AbstractArray, nev=1, tol::T=1e-13) where {T}
	n = length(q)
	function A(y::AbstractVector, x::AbstractVector)
		@inbounds y1 = view(y, 1:n); @inbounds y2 = view(y, n+1:2*n)
		@inbounds x1 = view(x, 1:n); @inbounds x2 = view(x, n+1:2*n)
		mul!(y1, P, x2)
		mul!(y2, P, x1)
		axpy!(-dot(q, x2)/r^2, q, y1)
		axpy!(-one(T), x1, y2)
	end
	D = LinearMap{T}(A, 2*n; ismutating=true)

	(λ, V, _, niter, nmult, _) = eigs(-D, [0*I C; C 0*I], nev=nev, which=:LR, tol=tol)
	return λ, V, niter, nmult
end