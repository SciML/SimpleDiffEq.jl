using SimpleDiffEq, OrdinaryDiffEq, StaticArrays, LinearAlgebra
function test(u, p, t)
    return -u
end

u0 = @SVector [1.0f0; 1.0f0; 1.0f0]
tspan = (0.0f0, 1.0f0)
p = @SVector [10.0f0, 28.0f0, 8 / 3.0f0]
prob = ODEProblem{false}(test, u0, tspan, p)

non_adaptive_algs = [GPUSimpleTsit5(), GPUSimpleVern7(), GPUSimpleVern9()]

adaptive_algs = [GPUSimpleATsit5(), GPUSimpleAVern7(), GPUSimpleAVern9()]

for (adaptive_alg, non_adaptive_alg) in zip(adaptive_algs, non_adaptive_algs)
    @info typeof(non_adaptive_alg)

    sol = solve(prob, non_adaptive_alg, dt = 0.01f0)
    asol = solve(
        prob, adaptive_alg, dt = 0.1f-1, abstol = 1.0f-7, reltol = 1.0f-7,
        save_everystep = false
    )

    @test sol.retcode == ReturnCode.Default
    @test asol.retcode == ReturnCode.Default

    ## Regression test

    bench_sol = solve(prob, Vern9(), adaptive = false, dt = 0.01f0)
    bench_asol = solve(
        prob, Vern9(), dt = 0.1f-1, save_everystep = false, abstol = 1.0f-7,
        reltol = 1.0f-7
    )

    @test norm(bench_sol.u[end] - sol.u[end]) < 5.0e-3
    @test norm(bench_asol.u - asol.u) < 5.0e-4

    ### solve parameters

    saveat = [0.0f0, 0.4f0]

    sol = solve(prob, non_adaptive_alg, dt = 0.01f0, saveat = saveat)

    asol = solve(
        prob, adaptive_alg, dt = 0.1f-1, abstol = 1.0f-7, reltol = 1.0f-7,
        saveat = saveat
    )

    bench_sol = solve(prob, Vern9(), adaptive = false, dt = 0.01f0, saveat = saveat)
    bench_asol = solve(
        prob, Vern9(), dt = 0.1f-1, save_everystep = false, abstol = 1.0f-7,
        reltol = 1.0f-7, saveat = saveat
    )

    @test norm(asol.u[end] - sol.u[end]) < 5.0e-3

    @test norm(bench_sol.u - sol.u) < 2.0e-4
    @test norm(bench_asol.u - asol.u) < 2.0e-4

    @test length(sol.u) == length(saveat)
    @test length(asol.u) == length(saveat)

    saveat = 0.0f0:0.01f0:1.0f0

    sol = solve(prob, non_adaptive_alg, dt = 0.01f0, saveat = saveat)

    asol = solve(
        prob, adaptive_alg, dt = 0.1f-1, save_everystep = false, abstol = 1.0f-7,
        reltol = 1.0f-7,
        saveat = saveat
    )

    bench_sol = solve(prob, Vern9(), adaptive = false, dt = 0.01f0, saveat = saveat)
    bench_asol = solve(
        prob, Vern9(), dt = 0.1f-1, save_everystep = false, abstol = 1.0f-7,
        reltol = 1.0f-7, saveat = saveat
    )

    @test norm(asol.u[end] - sol.u[end]) < 6.0e-3

    @test norm(bench_sol.u - sol.u) < 2.0e-3
    @test norm(bench_asol.u - asol.u) < 3.0e-3

    @test length(sol.u) == length(saveat)
    @test length(asol.u) == length(saveat)

    sol = solve(prob, non_adaptive_alg, dt = 0.01f0, save_everystep = false)

    bench_sol = solve(prob, Vern9(), adaptive = false, dt = 0.01f0, save_everystep = false)

    @test norm(bench_sol.u - sol.u) < 5.0e-3

    @test length(sol.u) == length(bench_sol.u)
end
