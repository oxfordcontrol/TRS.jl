using Test, Random
using LinearAlgebra
using MATLAB
using Arpack

rng = MersenneTwister(123)
for n in [2, 5, 30, 100, 1000]
    P = randn(rng, n, n); P = (P + P')/2;
    q = randn(rng, n)/10
    r = [1e-4 1e-2 1 1000]
    eye = Matrix{Float64}(I, n, n)
    for i = 1:length(r)
        if rand(rng) < .1
            # Make P positive definite
            P = P + 1.1*abs(eigs(P, nev=1, which=:SR)[1][1])*I;
        elseif rand(rng) < .1
            # Make P negative definite
            P = P - 1.1*abs(eigs(P, nev=1, which=:LR)[1][1])*I; 
        end
        if n < 30
            X, info = TRS.trs_small(P, q, r[i], compute_local=true)
        else
            X, info = TRS.trs(P, q, r[i], compute_local=true)
        end
        x_g = X[:, 1]
        x_matlab, λ_matlab = mxcall(:TRSgep, 2, P, q, eye, r[i])
        str = "Trs - r:"*string(r[i])
        @show norm(P*x_g + q + info.λ[1]*x_g), norm(P*x_matlab[:, 1] + q + λ_matlab*x_matlab[:, 1])
        # @show info.λ[1] + (dot(x_g, P*x_g) + dot(q,x_g))/r[end]^2
        # @show info.λ[1], λ_matlab
        @testset "$str" begin
            @test info.λ[1] - λ_matlab <= 1e-6*λ_matlab
            if size(x_matlab, 2) > 1
                diff = min(norm(x_g - x_matlab[:, 1]), norm(x_g - x_matlab[:, 2]))
            else
                diff = norm(x_g - x_matlab)
            end
            @test diff <= 1e-3*r[i]
        end
    end
    # hard case
    λ_min, v, _ = eigs(-P, nev=1, which=:LR)
    v = v/norm(v)
    q = (I - v*v')*q
    if n < 0
        X, info = TRS.trs_small(P, q, r[end], compute_local=true)
    else
        X, info = TRS.trs(P, q, r[end], compute_local=true)
    end
    x_g = X[:, 1]
    x_matlab, λ_matlab = mxcall(:TRSgep, 2, P, q, eye, r[end])
    @show norm(P*x_g + q + info.λ[1]*x_g), norm(P*x_matlab[:, 1] + q + λ_matlab*x_matlab[:, 1])
    # @show info.λ[1] + (dot(x_g, P*x_g) + dot(q,x_g))/r[end]^2
    # @show info.λ[1], λ_matlab
    @testset "Trs - hard case" begin
        @test info.λ[1] - λ_matlab <= 1e-6*λ_matlab
        if size(x_matlab, 2) > 1
            diff = min(norm(x_g - x_matlab[:, 1]), norm(x_g - x_matlab[:, 2]))
        else
            diff = norm(x_g - x_matlab)
        end
        @test diff <= 1e-3*r[end]
    end
end


nothing
