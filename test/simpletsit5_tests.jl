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
dt = 0.01
oop = SimpleDiffEq.simpletsit5_init(loop, false, SVector{3}(u0), 0.0, dt, [10, 28, 8 / 3])
step!(oop)
for i in 1:10000
    step!(oop)
    if isnan(oop.u[1]) || isnan(oop.u[2]) || isnan(oop.u[3])
        error("oop nan")
    end
end

iip = SimpleDiffEq.simpletsit5_init(liip, true, copy(u0), 0.0, dt, [10, 28, 8 / 3])
step!(iip)
for i in 1:10000
    step!(iip)
    if isnan(iip.u[1]) || isnan(iip.u[2]) || isnan(iip.u[3])
        error("iip nan")
    end
end

u0 = 10ones(3)
dt = 0.01

odeoop = ODEProblem{false}(loop, SVector{3}(u0), (0.0, 100.0), [10, 28, 8 / 3])
odeiip = ODEProblem{true}(liip, u0, (0.0, 100.0), [10, 28, 8 / 3])

oop = init(odeoop, SimpleTsit5(), dt = dt)
step!(oop);
step!(oop);

iip = init(odeiip, SimpleTsit5(), dt = dt)
step!(iip);
step!(iip);

deoop = DiffEqBase.init(odeoop, Tsit5(); adaptive = false,
    save_everystep = false, dt = dt)
step!(deoop);
step!(deoop);
@test oop.u == deoop.u

deiip = DiffEqBase.init(odeiip, Tsit5();
    adaptive = false, save_everystep = false,
    dt = dt)
step!(deiip);
step!(deiip);
@test iip.u≈deiip.u atol=1e-14

sol = solve(odeoop, SimpleTsit5(), dt = dt)

# https://github.com/SciML/SimpleDiffEq.jl/pull/72
f(u, p, t) = 1.01 * u
u0 = 1 / 2
tspan = (0.0, 1.0)
prob = ODEProblem(f, u0, tspan)
sol1 = solve(prob, SimpleTsit5(), dt = dt)

function ode(x, p, t)
    dx = sin(x[1])
    return ([dx])
end
prob = ODEProblem(ode, [1.0], (0.0, 0.05), nothing)
sol = solve(prob, SimpleTsit5(), dt = 0.05/11) # On my PC, the integration ends at 0.04545...
@test sol.t[end] == 0.05

#=
using BenchmarkTools

function bench()
    u0 = 10ones(3)
    dt = 0.01

    oop = SimpleDiffEq.simpletsit5_init(loop, false, SVector{3}(u0), 0.0, dt, [10, 28, 8/3])
    step!(oop)

    iip = SimpleDiffEq.simpletsit5_init(liip, true, u0, 0.0, dt, [10, 28, 8/3])
    step!(iip)

    odeoop = ODEProblem{false}(loop, SVector{3}(u0), (0.0, Inf),  [10, 28, 8/3])
    deoop = DiffEqBase.init(odeoop, Tsit5();
    adaptive = false, save_everystep = false, dt = dt, maxiters = Inf)
    step!(deoop)

    odeiip = ODEProblem{true}(liip, u0, (0.0, Inf),  [10, 28, 8/3])
    deiip = DiffEqBase.init(odeiip, Tsit5();
    adaptive = false, save_everystep = false, dt = dt, maxiters = Inf)

    step!(deiip)

    println("Minimal integrator times")
    println("In-place time")
    @btime step!($iip)

    println("Out of place time")
    @btime step!($oop)

    println("Standard Tsit5 times")
    println("In-place time")
    @btime step!($deiip)

    println("Out of place time")
    @btime step!($deoop)

end

bench()

function bench_only_step()
  u0 = 10ones(3)
  dt = 0.01
  oop = SimpleDiffEq.simpletsit5_init(loop, false, SVector{3}(u0), 0.0, dt, [10, 28, 8/3])
  DiffEqBase.step!(oop)
  iip = SimpleDiffEq.simpletsit5_init(liip, true, u0, 0.0, dt, [10, 28, 8/3])
  DiffEqBase.step!(iip)
  odeoop = ODEProblem{false}(loop, SVector{3}(u0), (0.0, Inf),  [10, 28, 8/3])
  deoop = DiffEqBase.init(odeoop, Tsit5();
  adaptive = false, save_everystep = false, dt = dt, maxiters = Inf)
  DiffEqBase.step!(deoop)
  odeiip = ODEProblem{true}(liip, u0, (0.0, Inf),  [10, 28, 8/3])
  deiip = DiffEqBase.init(odeiip, Tsit5();
  adaptive = false, save_everystep = false, dt = dt, maxiters = Inf)
  DiffEqBase.step!(deiip)
  println("Only Step times")
  println("In-place time")
  @btime DiffEqBase.step!($iip)
  println("Out of place time")
  @btime DiffEqBase.step!($oop)
  println("Standard Tsit5 times")
  println("In-place time")
  @btime OrdinaryDiffEq.perform_step!($deiip,$(deiip.cache))
  println("Out of place time")
  @btime OrdinaryDiffEq.perform_step!($deoop,$(deoop.cache))
end
bench_only_step()
=#
