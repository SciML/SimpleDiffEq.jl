using SimpleDiffEq, StaticArrays, Test

# Test interface compatibility with different number types

function decay(u, p, t)
    return -u
end

function decay!(du, u, p, t)
    du .= -u
    return nothing
end

@testset "BigFloat scalar support" begin
    u0 = BigFloat(1.0)
    tspan = (BigFloat(0.0), BigFloat(1.0))
    dt = BigFloat(0.01)
    prob = ODEProblem{false}(decay, u0, tspan, nothing)

    sol = solve(prob, SimpleEuler(), dt = dt)
    @test eltype(sol.u) == BigFloat

    sol = solve(prob, SimpleRK4(), dt = dt)
    @test eltype(sol.u) == BigFloat

    sol = solve(prob, SimpleTsit5(), dt = dt)
    @test eltype(sol.u) == BigFloat

    sol = solve(prob, SimpleATsit5(), dt = dt)
    @test eltype(sol.u) == BigFloat
end

@testset "BigFloat Vector support (OOP)" begin
    u0 = BigFloat[1.0, 2.0, 3.0]
    tspan = (BigFloat(0.0), BigFloat(1.0))
    dt = BigFloat(0.01)
    prob = ODEProblem{false}(decay, u0, tspan, nothing)

    sol = solve(prob, SimpleEuler(), dt = dt)
    @test eltype(sol.u[end]) == BigFloat

    sol = solve(prob, SimpleRK4(), dt = dt)
    @test eltype(sol.u[end]) == BigFloat

    sol = solve(prob, SimpleTsit5(), dt = dt)
    @test eltype(sol.u[end]) == BigFloat
end

@testset "BigFloat Vector support (IIP)" begin
    u0 = BigFloat[1.0, 2.0, 3.0]
    tspan = (BigFloat(0.0), BigFloat(1.0))
    dt = BigFloat(0.01)
    prob = ODEProblem{true}(decay!, u0, tspan, nothing)

    sol = solve(prob, SimpleEuler(), dt = dt)
    @test eltype(sol.u[end]) == BigFloat

    sol = solve(prob, SimpleRK4(), dt = dt)
    @test eltype(sol.u[end]) == BigFloat

    sol = solve(prob, SimpleTsit5(), dt = dt)
    @test eltype(sol.u[end]) == BigFloat
end

@testset "SVector{BigFloat} support" begin
    u0 = SVector{3, BigFloat}(BigFloat(1.0), BigFloat(2.0), BigFloat(3.0))
    tspan = (BigFloat(0.0), BigFloat(1.0))
    dt = BigFloat(0.01)
    prob = ODEProblem{false}(decay, u0, tspan, nothing)

    sol = solve(prob, SimpleEuler(), dt = dt)
    @test eltype(sol.u[end]) == BigFloat

    sol = solve(prob, SimpleRK4(), dt = dt)
    @test eltype(sol.u[end]) == BigFloat

    sol = solve(prob, SimpleTsit5(), dt = dt)
    @test eltype(sol.u[end]) == BigFloat

    # GPU-style solvers with SVector{BigFloat}
    sol = solve(prob, GPUSimpleTsit5(), dt = dt)
    @test eltype(sol.u[end]) == BigFloat

    sol = solve(prob, GPUSimpleATsit5(), dt = dt)
    @test eltype(sol.u[end]) == BigFloat
end
