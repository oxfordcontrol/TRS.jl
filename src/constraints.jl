function trs_boundary(P, q::AbstractVector{T}, r::T, A::AbstractMatrix{T}, b::AbstractVector{T}; kwargs...) where {T}
	F = factorize([I A'; A 0*I]) # KKT matrix
	n = length(q)
	x_ = zeros(size(F, 1))
	y_ = zeros(size(F, 1))
	function project!(x::AbstractVector{T})
		copyto!(view(x_, 1:n), x)
		ldiv!(y_, F, x_)
		copyto!(x, view(y_, 1:n))
	end
	x = (F\[zeros(n); b])[1:n] # x is the minimizer of ‖x‖ with Ax = b
	@assert(norm(x) <= r, "The problem is infeasible.")
	d = project!(randn(n)) # Find a direction in the nullspace of A
	# Calculate alpha such that ‖x + alpha*d‖ = r
	alpha = roots(Poly([norm(x)^2 - r^2, 2*d'*x, norm(d)^2]))
	@assert(isreal(alpha), "The problem is infeasible.")
	x += alpha[1]*d # Now ‖x‖ = r

	return trs_boundary(P, q, r, project!, x; kwargs...)
end

function trs_boundary(P, q::AbstractVector{T}, r::T, project!, x::AbstractVector{T}; kwargs...) where {T}
	"""
	Solves the TRS problem
	minimize    ½x'Px + q'x
	subject to  ‖x‖ ≤ r
				Ax = b.

	A and b are replaced by project! and x, where:
	- project!(x) project x to the nullspace of A; and
	- x is a point with ‖x‖ = r and Ax = b
	"""
	n = length(q)
	λ_max = max(eigs(P, nev=1, which=:LR)[1][1], 0)
	function p(y::AbstractVector{T}, x::AbstractVector{T}) where {T}
		mul!(y, P, x)
		axpy!(-λ_max, x, y)
		project!(y)
	end
	x0 = x - project!(copy(x))
	P_ = LinearMap{T}(p, n; ismutating=true, issymmetric=true)
	q_ = project!(q + P*x0 - λ_max*x0)
	r_ = norm(x - x0)
	output = trs_boundary((nev; kw...) -> eigenproblem(P_, q_, r_, nev;
			v0=[project!(randn(n)); project!(randn(n))], kw...,),
			(λ, V; kw...) -> pop_solution!(P_, q_, r_, I, V, λ; kw...); kwargs...)
	return shift_output(output..., x0, λ_max)
end

function shift_output(x1, x2, info, x0, λ_max)
	x1 .+= x0
	x2 .+= x0
	info.λ .-= λ_max
	return x1, x2, info
end

function shift_output(x1, info, x0, λ_max)
	x1 .+= x0
	info.λ .-= λ_max
	return x1, info
end
