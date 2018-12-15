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

# Test keywords:
oop = init(odeoop,SimpleATsit5(),dt=dt, reltol = 1e-9, abstol = 1e-9)
step!(oop); step!(oop)
deoop = DiffEqBase.init(odeoop, Tsit5(); dt = dt, reltol=1e-9, abstol=1e-9)
step!(deoop); step!(deoop)

@test oop.u ≈ deoop.u atol=1e-14
@test oop.t ≈ deoop.t atol=1e-14

###################################################################################
# Internal norm test:
function moop(u, p, t)
    x = loop(u[:, 1], p, t)
    y = loop(u[:, 2], p, t)
    return hcat(x,y)
end
function miip(du, u, p, t)
    @views begin
        liip(du[:, 1], u[:, 1], p, t)
        liip(du[:, 2], u[:, 2], p, t)
    end
    return nothing
end

ran = rand(SVector{3})
odemoop = ODEProblem{false}(moop, SMatrix{3,2}(hcat(u0, ran)), (0.0, 100.0),  [10, 28, 8/3])
odemiip = ODEProblem{true}(miip, hcat(u0, ran), (0.0, 100.0),  [10, 28, 8/3])

using LinearAlgebra

oop = init(odemoop,SimpleATsit5(),dt=dt, internalnorm = u -> norm(u[:, 1]))
step!(oop); step!(oop)

iip = init(odemiip,SimpleATsit5(),dt=dt, internalnorm = u -> norm(u[:, 1]))
step!(iip); step!(iip)

@test oop.u ≈ iip.u atol=1e-14
@test oop.t ≈ iip.t atol=1e-14

###################################################################################
# VectorVector test:
function vvoop(du, u, p, t)
    @inbounds for j in 1:2
        du[j] = loop(u[j], p, t)
    end
    return nothing
end
function vviip(du, u, p, t)
    @inbounds for j in 1:2
        liip(du[j], u[j], p, t)
    end
    return nothing
end

ran = rand(3)
odevoop = ODEProblem{true}(vvoop, [SVector{3}(u0),  SVector{3}(ran)], (0.0, 100.0),  [10, 28, 8/3])
odeviip = ODEProblem{true}(vviip, [u0, ran], (0.0, 100.0),  [10, 28, 8/3])

viip = init(odeviip,SimpleATsit5(),dt=dt; internalnorm = u -> SimpleDiffEq.defaultnorm(u[1]))
step!(viip); step!(viip)

iip = init(odeiip,SimpleATsit5(),dt=dt)
step!(iip); step!(iip)

@test iip.u ≈ viip.u[1] atol=1e-14
