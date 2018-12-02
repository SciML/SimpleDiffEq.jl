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
odeiip = ODEProblem{true}(liip, u0, (0.0, 100.0),  [10, 28, 8/3])

oop = init(odeoop,SimpleATsit5(),dt=dt)
step!(oop); step!(oop)

iip = init(odeiip,SimpleATsit5(),dt=dt)
step!(iip); step!(iip)

deoop = DiffEqBase.init(odeoop, Tsit5(); dt = dt)
step!(deoop); step!(deoop)
@test oop.u ≈ deoop.u atol=1e-14
@test oop.t ≈ deoop.t atol=1e-14

deiip = DiffEqBase.init(odeiip, Tsit5(); dt = dt)
step!(deiip); step!(deiip)
@test iip.u ≈ deiip.u atol=1e-9
@test iip.t ≈ deiip.t atol=1e-12

sol = solve(odeoop,SimpleATsit5(),dt=dt)
