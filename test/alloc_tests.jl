using SimpleDiffEq
using DiffEqBase
using StaticArrays
using Test

@testset "Allocation Tests - Zero Allocation Verification" begin
    # Define test problems (Lorenz system)
    function loop_oop(u, p, t)
        @inbounds begin
            σ = p[1]
            ρ = p[2]
            β = p[3]
            du1 = σ * (u[2] - u[1])
            du2 = u[1] * (ρ - u[3]) - u[2]
            du3 = u[1] * u[2] - β * u[3]
            return SVector{3}(du1, du2, du3)
        end
    end

    function loop_iip!(du, u, p, t)
        σ = p[1]
        ρ = p[2]
        β = p[3]
        du[1] = σ * (u[2] - u[1])
        du[2] = u[1] * (ρ - u[3]) - u[2]
        du[3] = u[1] * u[2] - β * u[3]
        return nothing
    end

    u0_vec = [10.0, 10.0, 10.0]
    u0_svec = SVector{3}(10.0, 10.0, 10.0)
    dt = 0.01
    p = [10.0, 28.0, 8 / 3]

    @testset "SimpleTsit5 step! - zero allocations" begin
        # OOP (out-of-place) with SVector
        integ_oop = SimpleDiffEq.simpletsit5_init(loop_oop, false, u0_svec, 0.0, dt, p)
        step!(integ_oop)  # warmup
        allocs = @allocated step!(integ_oop)
        @test allocs == 0

        # IIP (in-place) with Vector
        integ_iip = SimpleDiffEq.simpletsit5_init(loop_iip!, true, copy(u0_vec), 0.0, dt, p)
        step!(integ_iip)  # warmup
        allocs = @allocated step!(integ_iip)
        @test allocs == 0
    end

    @testset "SimpleRK4 step! - zero allocations" begin
        # OOP (out-of-place) with SVector
        integ_oop = SimpleDiffEq.simplerk4_init(loop_oop, false, u0_svec, 0.0, dt, p)
        step!(integ_oop)  # warmup
        allocs = @allocated step!(integ_oop)
        @test allocs == 0

        # IIP (in-place) with Vector
        integ_iip = SimpleDiffEq.simplerk4_init(loop_iip!, true, copy(u0_vec), 0.0, dt, p)
        step!(integ_iip)  # warmup
        allocs = @allocated step!(integ_iip)
        @test allocs == 0
    end

    @testset "SimpleEuler step! - zero allocations" begin
        # OOP (out-of-place) with SVector
        integ_oop = SimpleDiffEq.simpleeuler_init(loop_oop, false, u0_svec, 0.0, dt, p)
        step!(integ_oop)  # warmup
        allocs = @allocated step!(integ_oop)
        @test allocs == 0

        # IIP (in-place) with Vector
        integ_iip = SimpleDiffEq.simpleeuler_init(loop_iip!, true, copy(u0_vec), 0.0, dt, p)
        step!(integ_iip)  # warmup
        allocs = @allocated step!(integ_iip)
        @test allocs == 0
    end

    @testset "SimpleATsit5 step! - zero allocations" begin
        # OOP (out-of-place) with SVector
        prob_oop = ODEProblem{false}(loop_oop, u0_svec, (0.0, 100.0), p)
        integ_oop = DiffEqBase.__init(prob_oop, SimpleATsit5(), dt = dt)
        step!(integ_oop)  # warmup
        allocs = @allocated step!(integ_oop)
        @test allocs == 0

        # IIP (in-place) with Vector
        prob_iip = ODEProblem{true}(loop_iip!, copy(u0_vec), (0.0, 100.0), p)
        integ_iip = DiffEqBase.__init(prob_iip, SimpleATsit5(), dt = dt)
        step!(integ_iip)  # warmup
        allocs = @allocated step!(integ_iip)
        @test allocs == 0
    end
end
