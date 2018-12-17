include("../src/TRS.jl")
using Main.TRS
using Test, Random
using LinearAlgebra
using Arpack
using Polynomials

rng = MersenneTwister(123)
for n in [3, 5, 10, 30, 100, 1000]
    P = randn(rng, n, n); P = (P + P')/2;
    q = randn(rng, n)/10
    m = Int(floor(n/2));
    A = randn(rng, m, n)
    b = randn(rng, m)
    # C_ = randn(n, n); C_ = (C_ + C_')/2;
    # Make C positive definite
    # C_ -= 1.01*min(minimum(eigvals(C_)), 0)*I
    r = [1e-4 1e-2 1 10 100 1000]
    eye = Matrix{Float64}(I, n, n)

    P_ = zeros(0, 0)
    q_ = zeros(0)
    r_ = 0
    N = zeros(0, 0)
    x0 = zeros(0)
    for i = 1:length(r)
        if rand(rng) < .1
            # Make P positive definite
            shift = 1.1*abs(eigs(P, nev=1, which=:SR)[1][1])
            P = P + shift*I;
        elseif rand(rng) < .1
            # Make P negative definite
            shift = -1.1*abs(eigs(P, nev=1, which=:LR)[1][1])
            P = P + shift*I; 
        end

        N = nullspace(A)
        x = A\b
        d = N*(N'*randn(n))
        P_ = Symmetric(N'*P*N)

        if norm(x) <= r[i]
            alpha = roots(Poly([norm(x)^2 - r[i]^2, 2*d'*x, norm(d)^2]))
            x += alpha[1]*d

            x0 = x - N*(N'*x)
            q_ = N'*(q + P*x0)

            r_ = sqrt(r[i]^2 - norm(x0)^2)

            x_g, x_l, info = trs(P, q, r[i], A, b, compute_local=true)
            x_g_reduced, x_l_reduced, info_reduced = trs(P_, q_, r_, compute_local=true)
            str = "Constrained trs - r:"*string(r[i])
            @testset "$str" begin
                @test info.λ[1] - info_reduced.λ[1] <= 1e-6*abs(info_reduced.λ[1])
                @test norm(x_g - N*x_g_reduced - x0) <= 1e-3*r[i]
                if !isnan(info_reduced.λ[2])
                    @test info.λ[2] - info_reduced.λ[2] <= 1e-6*abs(info_reduced.λ[2])
                    @test norm(x_l - N*x_l_reduced - x0) <= 1e-3*r[i]
                end
            end
        end
    end
    # hard case
    λ_min, v, _ = eigs(-P, nev=1, which=:LR)
    v = v/norm(v)
    q = (I - v*v')*q
    q_ = N'*(q + P*x0)
    x_g, x_l, info = trs(P, q, r[end], A, b, compute_local=true)
    x_g_reduced, x_l_reduced, info_reduced = trs(P_, q_, r_, compute_local=true)
    @testset "Constrainted trs - hard case" begin
        @test info.λ[1] - info_reduced.λ[1] <= 1e-6*abs(info_reduced.λ[1])
        @test norm(x_g - N*x_g_reduced - x0) <= 1e-3*r[end]
        if !isnan(info_reduced.λ[2])
            @test info.λ[2] - info_reduced.λ[2] <= 1e-6*abs(info_reduced.λ[2])
            @test norm(x_l - N*x_l_reduced - x0) <= 1e-3*r[end]
        end
    end
end

nothing
