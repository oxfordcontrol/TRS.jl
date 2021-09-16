module TRSTests
using TRS

using Test, JLD2, MATLAB # [extras] 
using Arpack, Polynomials, MATLAB, LinearAlgebra, Random # [deps]

@testset "All Unit Tests" begin
  include("./trs.jl")
  include("./trs_constraints.jl")
  include("./trs_ellipsoid.jl")
  include("./special_cases.jl")
end

end # module