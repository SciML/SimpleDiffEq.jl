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
@test oop.u ≈ deoop.u atol=1e-5
@test oop.t ≈ deoop.t atol=1e-5

deiip = DiffEqBase.init(odeiip, Tsit5(); dt = dt)
step!(deiip); step!(deiip)
@test iip.u ≈ deiip.u atol=1e-5
@test iip.t ≈ deiip.t atol=1e-5

sol = solve(odeoop,SimpleATsit5(),dt=dt)

# Test keywords:
oop = init(odeoop,SimpleATsit5(),dt=dt, reltol = 1e-9, abstol = 1e-9)
step!(oop); step!(oop)
deoop = DiffEqBase.init(odeoop, Tsit5(); dt = dt, reltol=1e-9, abstol=1e-9)
step!(deoop); step!(deoop)

@test oop.u ≈ deoop.u atol=1e-5
@test oop.t ≈ deoop.t atol=1e-5

# Test reinit!
reinit!(oop, odeoop.u0; dt = dt)
reinit!(iip, odeiip.u0; dt = dt)
step!(oop); step!(oop)
step!(iip); step!(iip)

@test oop.u ≈ deoop.u atol=1e-5
@test oop.t ≈ deoop.t atol=1e-5
@test iip.u ≈ deiip.u atol=1e-5
@test iip.t ≈ deiip.t atol=1e-5

# Interpolation tests
uprev = copy(oop.u)
step!(oop)
@test uprev ≈ oop(oop.tprev) atol = 1e-12
@test oop(oop.t) ≈ oop.u atol = 1e-12

uprev = copy(iip.u)
step!(iip)
@test uprev ≈ iip(iip.tprev) atol = 1e-12
@test iip(iip.t) ≈ iip.u atol = 1e-12

# Interpolation tests comparing Tsit5 and SimpleATsit5
function f(du,u,p,t)
    du[1] = 2.0*u[1] + 3.0*u[2]
    du[2] = 4.0*u[1] + 5.0*u[2]
end
tmp = [1.0;1.0]
tspan = (0.0,1.0)
prob = ODEProblem(f,tmp,tspan)
integ1 = init(prob, SimpleATsit5(), abstol = 1e-6, reltol = 1e-6, save_everystep = false, dt = 0.1)
integ2 = init(prob, Tsit5(), abstol = 1e-6, reltol = 1e-6, save_everystep = false, dt = 0.1)
step!(integ2)
step!(integ1)
for i in 1:9
    x = i/10
    y = 1 - x
    @test integ1(x * integ2.t + y * integ2.tprev) ≈ integ2(x * integ2.t + y * integ2.tprev) atol=1e-7
end
step!(integ2)
step!(integ1)
for i in 1:9
    x = i/10
    y = 1 - x
    @test integ1(x * integ2.t + y * integ2.tprev) ≈ integ2(x * integ2.t + y * integ2.tprev) atol=1e-7
end
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

oop = init(odemoop,SimpleATsit5(),dt=dt, internalnorm = (u, t) -> norm(u[:, 1]))
step!(oop); step!(oop)

iip = init(odemiip,SimpleATsit5(),dt=dt, internalnorm = (u, t) -> norm(u[:, 1]))
step!(iip); step!(iip)

@test oop.u ≈ iip.u atol=1e-5
@test oop.t ≈ iip.t atol=1e-5

###################################################################################
# VectorVector test:
function vvoop(du, u, p, t) # takes Vector{SVector}
    @inbounds for j in 1:2
        du[j] = loop(u[j], p, t)
    end
    return nothing
end
function vviip(du, u, p, t) # takes Vector{Vector}
    @inbounds for j in 1:2
        liip(du[j], u[j], p, t)
    end
    return nothing
end

ran = rand(3)
odevoop = ODEProblem{true}(vvoop, [SVector{3}(u0),  SVector{3}(ran)], (0.0, 100.0),  [10, 28, 8/3])
odeviip = ODEProblem{true}(vviip, [u0, ran], (0.0, 100.0),  [10, 28, 8/3])

viip = init(odeviip,SimpleATsit5(),dt=dt; internalnorm = (u, t) -> DiffEqBase.ODE_DEFAULT_NORM(u[1], t))
step!(viip); step!(viip)

iip = init(odeiip,SimpleATsit5(),dt=dt)
step!(iip); step!(iip)

@test iip.u ≈ viip.u[1] atol=1e-5

voop = init(odevoop,SimpleATsit5(),dt=dt,internalnorm = (u, t) -> DiffEqBase.ODE_DEFAULT_NORM(u[1], t))
step!(voop); step!(voop)

oop = init(odeoop,SimpleATsit5(),dt=dt)
step!(oop); step!(oop)

@test oop.u ≈ voop.u[1] atol=1e-5

# Final test that the states of both methods should be the same:
@test voop.u[2] ≈ viip.u[2] atol=1e-5



# viip = init(odeviip,SimpleATsit5(),dt=dt; internalnorm = u -> SimpleDiffEq.defaultnorm(u[1]))
# step!(viip); step!(viip)
# x = Float64[]; y = Float64[]
# for i in 1:1000
#     step!(viip)
#     push!(x, viip.u[1][1])
#     push!(y, viip.u[1][2])
# end
# using PyPlot
# plot(x,y)

###################################################################################
# Issue #46 check (step!(integ, dt, true))
# see https://github.com/SciML/SimpleDiffEq.jl/issues/46
using SimpleDiffEq, OrdinaryDiffEq, StaticArrays
using PyPlot

@inbounds function duffing_rule(x, p, t)
    ω, f, d, β = p
    dx1 = x[2]
    dx2 = f*cos(ω*t) - β*x[1] - x[1]^3 - d * x[2]
    return SVector(dx1, dx2)
end
prob = ODEProblem(duffing_rule, SVector(0.1, 0.2), (0.0, 1e12), [0.3, 0.1, 0.2, -1])

T = 2π/0.3 # this is the period of the oscillator

integ2 = init(prob, SimpleATsit5(), reltol=1e-9)
step!(integ2, T*20, true)
v=zeros(200,2)
for k in 1:200
   step!(integ2, T, true)
   v[k,:] = integ2.u
end

@test length(unique(round.(v; digits = 3))) == 2
