using SimpleDiffEq
using DiffEqDevTools, Test, Random
using DiffEqProblemLibrary.ODEProblemLibrary: importodeproblems; importodeproblems()
import DiffEqProblemLibrary.ODEProblemLibrary: prob_ode_2Dlinear, prob_ode_2Dlinear
Random.seed!(1)
testTol = 0.2
prob = prob_ode_2Dlinear

@test_nowarn sol = solve(prob, SimpleTsit5(), dt=0.25)
dts = 1 .//2 .^(7:-1:3)
sim = test_convergence(dts,prob,SimpleTsit5())
@test abs(Float64(sim.𝒪est[:l2]-5)) < testTol+0.1
