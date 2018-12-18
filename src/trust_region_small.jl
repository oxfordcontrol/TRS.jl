function trs_small(P::AbstractMatrix{T}, q::AbstractVector{T}, r::T; kwargs...) where {T}
	output = trs_boundary_small(P, q, r; kwargs...)
	return check_interior!(output..., P, q)
end

function trs_small(P::AbstractMatrix{T}, q::AbstractVector{T}, r::T, C::AbstractMatrix{T}; kwargs...) where {T}
	output = trs_boundary_small(P, q, r, C; kwargs...)
	return check_interior!(output..., P, q)
end

function check_interior_small!(x1::AbstractVector, info::TRSinfo, P, q::AbstractVector)
	if info.λ[1] <= 0 # Global solution is in the interior
		x1 = -P\q
		info.λ[1] = 0
	end
	return x1, info
end

function check_interior_small!(x1::AbstractVector, x2::AbstractVector, info::TRSinfo, P, q::AbstractVector)
	if info.λ[1] <= 0 # Global solution is in the interior
		x1, info = check_interior_small!(x1, info, P, q)
	end
	if info.λ[2] <= 0
		# No local-no-global minimiser can exist in the interior
		x2 = []
		info.λ[2] = NaN
	end
	return x1, x2, info
end

function trs_boundary_small(P::AbstractMatrix{T}, q::AbstractVector{T}, r::T, C::AbstractMatrix{T}; kwargs...) where T
	check_inputs(P, q, r, C)
	return trs_boundary((nev; kw...) -> gen_eigenproblem_small(P, q, r, C, nev; kw...),
		   (λ, V; kw...) -> pop_solution_small!(P, q, r, C, V, λ; kw...); kwargs...)
end

function trs_boundary_small(P::AbstractMatrix{T}, q::AbstractVector{T}, r::T; kwargs...) where {T}
	check_inputs(P, q, r)
	return trs_boundary((nev; kw...) -> eigenproblem_small(P, q, r, nev; kw...),
		   (λ, V; kw...) -> pop_solution_small!(P, q, r, I, V, λ; kw...); kwargs...)
end

function pop_solution_small!(P::AbstractMatrix{T}, q::AbstractVector{T}, r::T, C, V, λ;
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

	norm_v1 = sqrt(dot(v1, C*v1))
	if norm_v1 >= tol_hard
		x1 = -sign(q'*v2)*r*v1/norm_v1
		x2 = zeros(0)
	else
		W = nullspace(Matrix(P) + l*I)
		y = -(P + l*C*(I + W*W'))\q
		α = roots(Poly([y'*(C*y) - r^2, 2*(C*v2)'*y, v2'*(C*v2)]))
		x1 = y + α[1]*v2
		x2 = y + α[2]*v2
	end

	return x1, x2, l
end