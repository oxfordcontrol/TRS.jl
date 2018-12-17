using Test, Random
using LinearAlgebra
using MATLAB
using Arpack

rng = MersenneTwister(123)
for n in [2, 5, 30, 100, 1000]
    P = randn(rng, n, n); P = (P + P')/2;
    q = randn(rng, n)/10
    C = randn(n, n); C = (C + C')/2;
    # Make C positive definite
    C -= 1.01*min(minimum(eigvals(C)), 0)*I
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
            x_g, x_l, info = trs_small(P, q, r[i], C, compute_local=true)
        else
            x_g, x_l, info = trs(P, q, r[i], C, compute_local=true)
        end
        x_matlab, λ_matlab = mxcall(:TRSgep, 2, P, q, C, r[i])
        # @show norm(P*x_g + q + info.λ[1]*C*x_g)
        # @show norm(P*x_matlab + q + λ_matlab*C*x_matlab)
        # @show info.λ[1], λ_matlab
        # @show info
        str = "Ellipsoidal Trs - r:"*string(r[i])
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
    if n < 30
        x_g, x_l, info = trs_small(P, q, r[end], compute_local=true)
    else
        x_g, x_l, info = trs(P, q, r[end], compute_local=true)
    end
    x_matlab, λ_matlab = mxcall(:TRSgep, 2, P, q, eye, r[end])
    @testset "Ellipsoidal Trs - hard case" begin
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
