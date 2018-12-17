using Test

@testset "All Unit Tests" begin
  include("./trs.jl")
  include("./trs_constraints.jl")
  include("./trs_ellipsoid.jl")
end
nothing