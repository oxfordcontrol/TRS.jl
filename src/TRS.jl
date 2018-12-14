# __precompile__(true)

module TRS

using LinearAlgebra
using LinearMaps
using IterativeSolvers
using Arpack
using Polynomials

include("trust_region.jl")
export trs, trs_boundary

end