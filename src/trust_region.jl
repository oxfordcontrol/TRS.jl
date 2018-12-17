function trs(P, q::AbstractVector{T}, r::T; kwargs...) where {T}
	output = trs_boundary(P, q, r; kwargs...)
	return check_interior!(output..., P, q)
end

function trs(P, q::AbstractVector{T}, r::T, C::AbstractArray{T}; kwargs...) where {T}
	output = trs_boundary(P, q, r, C; kwargs...)
	return check_interior!(output..., P, q)
end

function check_interior!(x1::AbstractVector, info::TRSinfo, P, q::AbstractVector)
	if info.λ[1] <= 0 # Global solution is in the interior
		cg!(x1, P, -q)
		info.λ[1] = 0
	end
	return x1, info
end

function check_interior!(x1::AbstractVector, x2::AbstractVector, info::TRSinfo, P, q::AbstractVector)
	if info.λ[1] <= 0 # Global solution is in the interior
		x1, info = check_interior!(x1, info, P, q)
	end
	if info.λ[2] <= 0
		# No local-no-global minimiser can exist in the interior
		x2 = []
		info.λ[2] = NaN
	end
	return x1, x2, info
end