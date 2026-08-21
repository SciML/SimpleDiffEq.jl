using SimpleDiffEq, DiffEqBase, Test

f(u, p, t) = 1.01 * u

@testset "Precompile workload" begin
    prob = ODEProblem(f, 0.5, (0.0, 1.0))
    sol = solve(prob, SimpleRK4(); dt = 0.1)

    @test sol.t[end] == 1.0
    @test sol.u[1] == 0.5
    @test sol.u[end] > sol.u[1]
end
