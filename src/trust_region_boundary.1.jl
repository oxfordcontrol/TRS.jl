mutable struct TRSinfo
	hard_case::Bool  # Flag indicating if we are in the hard case
	niter::Int # Number of iterations in eigs
	nmul::Int  # Number of multiplication with P in eigs
	λ::Vector  # Lagrange Multiplier(s)

	function TRSinfo(hard_case::Bool, niter::Int, nmul::Int, λ::Vector)
		new(hard_case, niter, nmul, λ)
	end
end

function trs_boundary(P, q::AbstractVector{T}, r::T, C::AbstractMatrix{T}; kwargs...) where {T}
	check_inputs(P, q, r, C)
	return trs_boundary((nev; kw...) -> gen_eigenproblem(P, q, r, C, nev; kw...),
		   (λ, V; kw...) -> pop_solution!(λ, V, P, q, r, C; kw...); kwargs...)
end

function trs_boundary(P, q::AbstractVector{T}, r::T; kwargs...) where {T}
	check_inputs(P, q, r)
	return trs_boundary((nev; kw...) -> eigenproblem(P, q, r, nev; kw...),
		   (λ, V; kw...) -> pop_solution!(λ, V, P, q, r, I; kw...); kwargs...)
end

function trs_boundary(solve_eigenproblem::Function, pop_solution!::Function;
	compute_local=false, tol_hard=2e-7, kwargs...)

	if compute_local
		nev=2  # We will need the two rightmost eigenvalues.
	else
		nev=1  # We will only need the rightmost eigenvalue
	end

	l, V, niter, nmult = solve_eigenproblem(nev; kwargs...)
	X, λ, hard_case = pop_solution!(l, V; tol_hard=tol_hard) # Pop global minimizer(s).
	if !compute_local || hard_case
		return X, TRSinfo(hard_case, niter, nmult, λ)
	else
		X_local, λ_local, _ = pop_solution!(l, V; tol_hard=tol_hard) # Pop local-no-global minimizer.
		return [X X_local], TRSinfo(hard_case, niter, nmult, [λ; λ_local])
	end
end

function check_inputs(P, q::AbstractVector{T}, r::T) where {T}
	@assert(issymmetric(P), "The cost matrix must be symmetric.")
	@assert(eltype(P) == T, "Inconsistent element types.")
	@assert(size(P, 1) == size(P, 2) == length(q), "Inconsistent matrix dimensions.")
end

function check_inputs(P, q::AbstractVector{T}, r::T, C::AbstractMatrix{T}) where {T}
	check_inputs(P, q, r)
	@assert(issymmetric(C), "The norm must be defined by a symmetric positive definite matrix.")
end

function pop_solution!(λ, V, P, q::AbstractVector{T}, r::T, C; tol_hard, direct=false) where {T}
	idx = argmax(real(λ)) 	# Pop rightmost eigenvector
	is_first_pop = all(isfinite.(λ))
	n = length(q)
	if abs(imag(λ[idx])) >= 1e-9 && !is_first_pop # A solution exists only on real eigenvalues
		return zeros(T, n, 0), zeros(T, 0), false
	end
	l = real(λ[idx]);
	complex_v = view(V, :, idx)
	if norm(real(complex_v)) > norm(imag(complex_v))
		v = real(complex_v)
	else
		v = imag(complex_v)  # Sometimes the retuned eigenvector is complex
	end
	v ./= norm(v)
	v1 = view(v, 1:n); v2 = view(v, n+1:2*n)

	# Extract solution
	norm_v1 = sqrt(dot(v1, C*v1))
	X = zeros(T, n, 0)
	if norm_v1 >= tol_hard || !is_first_pop
		hard_case = false
		x = -r*v1/norm_v1
		if sign(q'*v2) < 0
			x .= -x
		end
		X = reshape(x, n, 1)
	else # hard case
		hard_case = true
		if !direct
			x1, x2 = extract_solution_hard_case(P, q, r, C, l, v1, v2, tol_hard)
		else
			x1, x2 = extract_solution_hard_case_direct(P, q, r, C, l, v1, v2)
		end
		# Sometimes extract_solution_hard_case fails in this case only return the Lagrange multipliers
		if isreal(x1) && isreal(x2)
			X = [x1 x2]
		end
	end
	λ[idx] = -Inf  # This ensures that the next pop_solution! would not consider the same solution.

	return X, l*ones(size(X, 2)), hard_case
end

function extract_solution_hard_case(P, q::AbstractVector{T}, r::T, C, l::T,
	v1::AbstractVector{T}, v2::AbstractVector{T}, tol::T) where T

	n = length(q)
	y, residual = cg_hard_case(P, q, C, l, reshape(v1/norm(v1), n, 1))
	nullspace_dim = 3
	while residual >= tol*norm(q) && nullspace_dim <= min(12, length(y))
		# Start on the range of P - that's important for constrained cases
		κ, W, _ = eigs(P, nev=nullspace_dim, which=:SR, v0 = P*randn(n))
		l = -minimum(κ)
		v2 = W[:, argmin(κ)]
		y, residual = cg_hard_case(P, q, C, l, W[:, abs.(κ .+ l) .< 1e-6])
		nullspace_dim *= 2
	end
	α = roots(Poly([y'*(C*y) - r^2, 2*(C*v2)'*y, v2'*(C*v2)]))
	if !isreal(α)
		@warn "Indirect extraction of solution failed; trying direct extraction."
		return extract_solution_hard_case_direct(P, q, r, C, l, v1, v2)
	end
	x1 = y + α[1]*v2
	x2 = y + α[2]*v2

	return x1, x2
end

function extract_solution_hard_case_direct(P, q::AbstractVector{T}, r::T, C, l::T,
	v1::AbstractVector{T}, v2::AbstractVector{T}) where T

	n = length(q)
	λ, W = eigen(P + l*C)
	W = W[:, abs.(λ) .<= 1e-9] # ToDo: Avoid a fixed tolerance
	y = -(Symmetric(P + l*C + C*W*W')\q)
	α = roots(Poly([y'*(C*y) - r^2, 2*(C*v2)'*y, v2'*(C*v2)]))
	x1 = y + α[1]*v2
	x2 = y + α[2]*v2

	return x1, x2
end

function cg_hard_case(P, q::AbstractVector{T}, C, λ::T, W::AbstractMatrix{T}) where {T}
	n = length(q)
	D = LinearMap{T}((x) -> P*x + λ*(C*x + W*(W'*x)), n; issymmetric=true)
 	# Start on the range of P - that's important for constrained cases
	y = cg!(P*randn(n), -D, q, tol=(eps(real(eltype(q)))/2)^(2/3))
	return y, norm(P*y + λ*(C*y) + q)
end