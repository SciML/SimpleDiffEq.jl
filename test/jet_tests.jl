using SimpleDiffEq
using JET
using DiffEqBase
using Test

@testset "JET Static Analysis" begin
    # Define test problems
    f_scalar(u, p, t) = 1.01 * u
    u0_scalar = 1.5
    tspan = (0.0, 1.0)
    prob_scalar = ODEProblem(f_scalar, u0_scalar, tspan)

    function f_iip!(du, u, p, t)
        du[1] = 1.01 * u[1]
        du[2] = 2.0 * u[2]
    end
    u0_iip = [1.5, 2.0]
    prob_iip = ODEProblem(f_iip!, u0_iip, tspan)

    @testset "SimpleEuler type stability" begin
        # OOP scalar
        integ_oop = DiffEqBase.__init(prob_scalar, SimpleEuler(), dt = 0.1)
        rep = JET.report_opt(DiffEqBase.step!, (typeof(integ_oop),))
        @test length(JET.get_reports(rep)) == 0

        # IIP array
        integ_iip = DiffEqBase.__init(prob_iip, SimpleEuler(), dt = 0.1)
        rep = JET.report_opt(DiffEqBase.step!, (typeof(integ_iip),))
        @test length(JET.get_reports(rep)) == 0
    end

    @testset "SimpleRK4 type stability" begin
        # OOP scalar
        integ_oop = DiffEqBase.__init(prob_scalar, SimpleRK4(), dt = 0.1)
        rep = JET.report_opt(DiffEqBase.step!, (typeof(integ_oop),))
        @test length(JET.get_reports(rep)) == 0

        # IIP array
        integ_iip = DiffEqBase.__init(prob_iip, SimpleRK4(), dt = 0.1)
        rep = JET.report_opt(DiffEqBase.step!, (typeof(integ_iip),))
        @test length(JET.get_reports(rep)) == 0
    end

    @testset "SimpleTsit5 type stability" begin
        # OOP scalar
        integ_oop = DiffEqBase.__init(prob_scalar, SimpleTsit5(), dt = 0.1)
        rep = JET.report_opt(DiffEqBase.step!, (typeof(integ_oop),))
        @test length(JET.get_reports(rep)) == 0

        # IIP array
        integ_iip = DiffEqBase.__init(prob_iip, SimpleTsit5(), dt = 0.1)
        rep = JET.report_opt(DiffEqBase.step!, (typeof(integ_iip),))
        @test length(JET.get_reports(rep)) == 0
    end

    @testset "SimpleATsit5 type stability" begin
        # OOP scalar
        integ_oop = DiffEqBase.__init(prob_scalar, SimpleATsit5(), dt = 0.1)
        rep = JET.report_opt(DiffEqBase.step!, (typeof(integ_oop),))
        @test length(JET.get_reports(rep)) == 0

        # IIP array
        integ_iip = DiffEqBase.__init(prob_iip, SimpleATsit5(), dt = 0.1)
        rep = JET.report_opt(DiffEqBase.step!, (typeof(integ_iip),))
        @test length(JET.get_reports(rep)) == 0
    end
end
