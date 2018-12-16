# __precompile__(true)

module TRS

using LinearAlgebra
using LinearMaps
using IterativeSolvers
using Arpack
using Polynomials

include("eigenproblems.jl")
include("trust_region_boundary.jl")
include("trust_region.jl")
export trs, trs_boundary

end