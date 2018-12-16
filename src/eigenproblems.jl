function eigenproblem(P, q::AbstractVector{T}, r::T, nev=1;
	tol=0.0, maxiter=300, v0=zeros((0,))) where {T}
	"""
	Calculates rightmost eigenvalues of

	|-P   qq'/r^2|  |v1|  =  λ|v1|
	| I        -P|  |v2|      |v2|

	with Arpack's eigs.
	"""
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

function gen_eigenproblem(P, q::AbstractVector{T}, r::T, C::AbstractArray, nev=1;
	tol=0.0, maxiter=300, v0=zeros((0,))) where {T}
	"""
	Calculates rightmost eigenvalues of

	|-P    qq'/r^2|  |v1|  =  λ |C   0| |v1|
	| C         -P|  |v2|       |0   C| |v2|

	with Arpack's eigs.
	"""
	n = length(q)
	function A(y::AbstractVector, x::AbstractVector)
		@inbounds y1 = view(y, 1:n); @inbounds y2 = view(y, n+1:2*n)
		@inbounds x1 = view(x, 1:n); @inbounds x2 = view(x, n+1:2*n)
		#=
		y1 .= -P*x1 + q*dot(q,x2)/r^2;
		y2 .= C*x1 - P*x2
		return y
		=#
		mul!(y1, C, x1)
		mul!(y2, P, x2)
		axpy!(-one(T), y1, y2)

		mul!(y1, P, x1)
		axpy!(-dot(q, x2)/r^2, q, y1)
	end
	D = LinearMap{T}(A, 2*n; ismutating=true)

	(λ, V, _, niter, nmult, _) = eigs(-D, [C 0*I; 0*I C], nev=nev, which=:LR, tol=tol)
	return λ, V, niter, 2*nmult
end