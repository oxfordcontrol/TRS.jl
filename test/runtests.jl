include("../src/TRS.jl")
using Main.TRS
using Test


@testset "All Unit Tests" begin
  include("./trs.jl")
  include("./trs_constraints.jl")
  include("./trs_ellipsoid.jl")
  include("./special_cases.jl")
end
nothing