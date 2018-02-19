using SimpleDiffEq, Base.Test

# dX_t = 2u dt + dW_t
f(u,p,t) = 2u
g(u,p,t) = 1
u0 = 0.5
tspan = (0.0,1.0)
prob = SDEProblem(f,g,u0,tspan)

sol = solve(prob,SimpleEM(),dt=0.25)

@test sol.t == collect(0:0.25:1.0)
@test length(sol.u) == 5
@test typeof(sol) <: DESolution
