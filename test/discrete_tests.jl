using SimpleDiffEq, Base.Test

prob = DiscreteProblem(0.5,(0.0,1.0))
sol =solve(prob,SimpleFunctionMap())

@test sol[1] == sol[end]

integrator = init(prob,SimpleFunctionMap())
step!(integrator)

@test integrator.u == integrator.uprev
@test integrator.t == 1.0

prob2 = DiscreteProblem(rand(4,2),(0.0,1.0))
sol = solve(prob2,SimpleFunctionMap())

@test sol[1] == sol[end]

integrator = init(prob,SimpleFunctionMap())
step!(integrator)

@test integrator.u == integrator.uprev
@test integrator.t == 1.0

prob = DiscreteProblem((u,p,t)->1.01u,0.5,(0.0,1.0))
sol = solve(prob,SimpleFunctionMap())

@test sol[end] ≈ .505

integrator = init(prob,SimpleFunctionMap())
step!(integrator)

@test integrator.u ≈ .505
@test integrator.uprev ≈ 0.5
@test integrator.t == 1.0

prob = DiscreteProblem((du,u,p,t)->du .= 1.01.*u,rand(4,2),(0.0,1.0))
sol = solve(prob,SimpleFunctionMap())

@test sol[end]./sol[1] ≈ fill(1.01,4,2)

integrator = init(prob,SimpleFunctionMap())
step!(integrator)

@test integrator.u ≈ prob.u0*1.01
@test integrator.uprev ≈ prob.u0
@test integrator.t == 1.0

@test_broken step!(integrator)
