# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
# Test related to the Euler solver
#
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

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

# Stability of the algorithm
# ==============================================================================
#
# In this case, the algorithm must solve the Lorenz Atractor problem without
# `NaN` in the solution vector.

# Out-of-place version of the algorithm
# -------------------------------------

u0 = 10ones(3)
dt = 0.01
oop = SimpleDiffEq.simpleeuler_init(loop,
    false,
    SVector{3}(u0),
    0.0,
    dt,
    [10, 28, 8 / 3])
step!(oop)

for i in 1:10000
    step!(oop)

    if isnan(oop.u[1]) || isnan(oop.u[2]) || isnan(oop.u[3])
        error("oop nan")
    end
end

# In-place version of the algorithm
# ---------------------------------

iip = SimpleDiffEq.simpleeuler_init(liip,
    true,
    copy(u0),
    0.0,
    dt,
    [10, 28, 8 / 3])

step!(iip)

for i in 1:10000
    step!(iip)

    if isnan(iip.u[1]) || isnan(iip.u[2]) || isnan(iip.u[3])
        error("iip nan")
    end
end

# Compare with the results obtained from OrdinaryDiffEq.jl
# ==============================================================================

# Out-of-place version of the algorithm
# --------------------------------------

u0 = 10ones(3)
dt = 0.01

odeoop = ODEProblem{false}(loop,
    SVector{3}(u0),
    (0.0, 100.0),
    [10, 28, 8 / 3])

oop = init(odeoop, SimpleEuler(), dt = dt)
step!(oop)
step!(oop)

deoop = DiffEqBase.init(odeoop,
    Euler();
    adaptive = false,
    save_everystep = false,
    dt = dt)
step!(deoop)
step!(deoop)

@test oop.u == deoop.u

sol = solve(odeoop, SimpleEuler(), dt = dt)

@test oop.u == sol.u[3]

sol = solve(odeoop, LoopEuler(), dt = dt)

@test oop.u == sol.u[3]

# In-place version of the algorithm
# ---------------------------------

odeiip = ODEProblem{true}(liip,
    u0,
    (0.0, 100.0),
    [10, 28, 8 / 3])

iip = init(odeiip, SimpleEuler(), dt = dt)
step!(iip)
step!(iip)

deiip = DiffEqBase.init(odeiip,
    Euler();
    adaptive = false,
    save_everystep = false,
    dt = dt)
step!(deiip)
step!(deiip)

@test iip.u == deiip.u

sol = solve(odeiip, SimpleEuler(), dt = dt)

@test iip.u == sol.u[3]

sol = solve(odeiip, LoopEuler(), dt = dt)

@test iip.u == sol.u[3]

# https://github.com/SciML/SimpleDiffEq.jl/pull/72
f(u, p, t) = 1.01 * u
u0 = 1 / 2
tspan = (0.0, 1.0)
prob = ODEProblem(f, u0, tspan)
sol1 = solve(prob, SimpleEuler(), dt = dt)
