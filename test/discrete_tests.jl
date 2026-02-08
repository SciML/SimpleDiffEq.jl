using SimpleDiffEq, Test

prob = DiscreteProblem(0.5, (0.0, 1.0))
sol = solve(prob, SimpleFunctionMap())

@test sol.u[1] == sol.u[end]

integrator = init(prob, SimpleFunctionMap())
step!(integrator)

@test integrator.u == integrator.uprev
@test integrator.t == 1.0

prob2 = DiscreteProblem(rand(4, 2), (0.0, 1.0))
sol = solve(prob2, SimpleFunctionMap())

@test sol.u[1] == sol.u[end]

integrator = init(prob, SimpleFunctionMap())
step!(integrator)

@test integrator.u == integrator.uprev
@test integrator.t == 1.0

prob = DiscreteProblem((u, p, t) -> 1.01u, 0.5, (0.0, 1.0))
sol = solve(prob, SimpleFunctionMap())

@test sol.u[end] ≈ 0.505

integrator = init(prob, SimpleFunctionMap())
step!(integrator)

@test integrator.u ≈ 0.505
@test integrator.uprev ≈ 0.5
@test integrator.t == 1.0

prob = DiscreteProblem((du, u, p, t) -> du .= 1.01 .* u, rand(4, 2), (0.0, 1.0))
sol = solve(prob, SimpleFunctionMap())

@test sol.u[end] ./ sol.u[1] ≈ fill(1.01, 4, 2)

integrator = init(prob, SimpleFunctionMap())
step!(integrator)

@test integrator.u ≈ prob.u0 * 1.01
@test integrator.uprev ≈ prob.u0
@test integrator.t == 1.0

@test_broken step!(integrator)
