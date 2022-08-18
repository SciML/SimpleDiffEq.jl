using SimpleDiffEq, StaticArrays, OrdinaryDiffEq, Test

function loop(u, p, t)
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
function liip(du, u, p, t)
    σ = p[1]
    ρ = p[2]
    β = p[3]
    du[1] = σ * (u[2] - u[1])
    du[2] = u[1] * (ρ - u[3]) - u[2]
    du[3] = u[1] * u[2] - β * u[3]
    return nothing
end

u0 = 10ones(3)
dt = 1e-2

odeoop = ODEProblem{false}(loop, SVector{3}(u0), (0.0, 100.0), [10, 28, 8 / 3])
sol = solve(odeoop, SimpleATsit5(), dt = dt)
sol2 = solve(odeoop, GPUSimpleATsit5(), dt = dt, abstol = 1e-6, reltol = 1e-3)
sol3 = solve(odeoop, GPUSimpleATsit5(), dt = dt, abstol = 1e-6, reltol = 1e-3,
             save_everystep = false)

@test sol.u[5] ≈ sol2.u[5]
@test sol.t[5] ≈ sol2.t[5]

@test sol.t[end] ≈ sol3.t[end]
@test sol2.t[end] ≈ sol3.t[end]

sol = solve(odeoop, Tsit5(), dt = dt, saveat = 0.0:0.1:100.0)
sol2 = solve(odeoop, GPUSimpleATsit5(), dt = dt, saveat = 0.0:0.1:100.0, abstol = 1e-6,
             reltol = 1e-3)
sol3 = solve(odeoop, SimpleATsit5(), dt = dt, saveat = 0.0:0.1:100.0)

@test sol[20]≈sol2[20] atol=1e-5
@test sol2.u[20] ≈ sol3.u[20]
@test sol.t ≈ sol2.t

dt = 1e-1

sol = solve(odeoop, SimpleTsit5(), dt = dt)
sol2 = solve(odeoop, GPUSimpleTsit5(), dt = dt)
sol3 = solve(odeoop, GPUSimpleTsit5(), dt = dt, save_everystep = false)
sol4 = solve(odeoop, GPUSimpleTsit5(), dt = dt, saveat = [5.0, 100.0])

@test sol.u ≈ sol2.u
@test sol.t ≈ sol2.t

@test sol.u[end] ≈ sol3.u[end]
@test sol2.u[end] ≈ sol3.u[end]

@test sol.u[end] ≈ sol4.u[end]
@test sol2.u[end] ≈ sol4.u[end]
@test sol(5.0) ≈ sol4.u[1]


#=
Solution seems to be sensitive at 100s,
hence changing the final tspan to test with save_everystep = false
=#
odeoop = remake(odeoop; tspan = (0.0, 10.0))

sol = solve(odeoop, Tsit5(), reltol = 1e-9, abstol = 1e-9, save_everystep = false)
sol1 = solve(odeoop, GPUSimpleATsit5(), reltol = 1e-9, abstol = 1e-9,
             save_everystep = false)

@test sol.u≈sol1.u atol=1e-5
@test sol.t ≈ sol1.t
