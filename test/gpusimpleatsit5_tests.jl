using SimpleDiffEq, StaticArrays, OrdinaryDiffEq, Test

function loop(u, p, t)
    @inbounds begin
        σ = p[1]; ρ = p[2]; β = p[3]
        du1 = σ*(u[2]-u[1])
        du2 = u[1]*(ρ-u[3]) - u[2]
        du3 = u[1]*u[2] - β*u[3]
        return SVector{3}(du1, du2, du3)
    end
end
function liip(du, u, p, t)
    σ = p[1]; ρ = p[2]; β = p[3]
    du[1] = σ*(u[2]-u[1])
    du[2] = u[1]*(ρ-u[3]) - u[2]
    du[3] = u[1]*u[2] - β*u[3]
    return nothing
end

u0 = 10ones(3)
dt = 1e-2

odeoop = ODEProblem{false}(loop, SVector{3}(u0), (0.0, 100.0),  [10, 28, 8/3])
sol  = solve(odeoop,SimpleATsit5()   ,dt=dt)
sol2 = solve(odeoop,GPUSimpleATsit5(),dt=dt)

@test sol.u == sol2.u
@test sol.t == sol2.t

sol  = solve(odeoop,Tsit5()          ,dt=dt,saveat=0.0:0.1:100.0)
sol2 = solve(odeoop,GPUSimpleATsit5(),dt=dt,saveat=0.0:0.1:100.0)

@test sol[20] ≈ sol2[20]
@test sol.t == sol2.t

sol  = solve(odeoop,SimpleTsit5()   ,dt=dt)
sol2 = solve(odeoop,GPUSimpleTsit5(),dt=dt)

@test sol.u == sol2.u
@test sol.t == sol2.t
