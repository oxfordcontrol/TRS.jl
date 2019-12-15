using Test
using JLD2

@load "special_case_1.jld2"

@testset "q=0, P semidefintie" begin
    for i = 1:100
        solution, info = trs(P, q, r)
        @test all(.!isnan.(solution))
    end
end

nothing
