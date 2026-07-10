# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
# Euler solver
#
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

"""
    SimpleEuler()

Construct a fixed-step forward Euler algorithm for `ODEProblem`s.

`SimpleEuler` advances ordinary differential equations with first-order explicit
Euler steps. It supports in-place and out-of-place problem functions and the
SciML integrator interface.

# Arguments

No positional arguments are accepted.

# Keywords

No constructor keywords are accepted. Pass `dt` to `solve` or
`init`.

# Returns

A `SimpleEuler` algorithm object for use with `ODEProblem`.

# Example

```julia
using SimpleDiffEq

f(u, p, t) = -0.5 * u

u0 = 1.0
tspan = (0.0, 10.0)
prob = ODEProblem(f, u0, tspan)

sol = solve(prob, SimpleEuler(), dt = 0.1)
```

# Notes

`dt` is required. Use `init` and `step!` for manual stepping or
dense interpolation over the most recent step.

# See Also

[`LoopEuler`](@ref), [`SimpleRK4`](@ref)
"""
struct SimpleEuler <: AbstractSimpleDiffEqODEAlgorithm end
export SimpleEuler

mutable struct SimpleEulerIntegrator{IIP, S, T, P, F} <:
    SciMLBase.AbstractODEIntegrator{SimpleEuler, IIP, S, T}
    f::F             # ..................................... Equations of motion
    uprev::S         # .......................................... Previous state
    u::S             # ........................................... Current state
    tmp::S           #  Auxiliary variable similar to state to avoid allocations
    tprev::T         # ...................................... Previous time step
    t::T             # ....................................... Current time step
    t0::T            # ........... Initial time step, only for re-initialization
    dt::T            # ............................................... Step size
    tdir::T          # ...................................... Not used for Euler
    p::P             # .................................... Parameters container
    u_modified::Bool # ..... If `true`, then the input of last step was modified
end

const SEI = SimpleEulerIntegrator

# If `true`, then the equation of motion format is `f!(du,u,p,t)` instead of
# `du = f(u,p,t)`.
DiffEqBase.isinplace(::SEI{IIP}) where {IIP} = IIP

################################################################################
#                                Initialization
################################################################################

function SciMLBase.__init(
        prob::ODEProblem, alg::SimpleEuler;
        dt = error("dt is required for this algorithm"), kwargs...
    )
    return simpleeuler_init(
        prob.f,
        DiffEqBase.isinplace(prob),
        prob.u0,
        prob.tspan[1],
        dt,
        prob.p
    )
end

function SciMLBase.__solve(
        prob::ODEProblem, alg::SimpleEuler;
        dt = error("dt is required for this algorithm"), kwargs...
    )
    u0 = prob.u0
    tspan = prob.tspan
    ts = Array(tspan[1]:dt:tspan[2])
    n = length(ts)
    us = Vector{typeof(u0)}(undef, n)

    @inbounds us[1] = _copy(u0)

    integ = simpleeuler_init(
        prob.f, DiffEqBase.isinplace(prob), prob.u0,
        prob.tspan[1], dt, prob.p
    )

    for i in 1:(n - 1)
        step!(integ)
        us[i + 1] = _copy(integ.u)
    end

    sol = SciMLBase.build_solution(prob, alg, ts, us, calculate_error = false)

    SciMLBase.has_analytic(prob.f) &&
        SciMLBase.calculate_solution_errors!(
        sol;
        timeseries_errors = true,
        dense_errors = false
    )

    return sol
end

@inline function simpleeuler_init(
        f::F, IIP::Bool, u0::S, t0::T, dt::T,
        p::P
    ) where
    {F, P, T, S}
    integ = SEI{IIP, S, T, P, F}(
        f,
        _copy(u0),
        _copy(u0),
        _copy(u0),
        t0,
        t0,
        t0,
        dt,
        sign(dt),
        p,
        true
    )

    return integ
end

################################################################################
#                                   Stepping
################################################################################

@inline @muladd function DiffEqBase.step!(integ::SEI{true, S, T}) where {T, S}
    integ.uprev .= integ.u
    tmp = integ.tmp
    f! = integ.f
    p = integ.p
    t = integ.t
    dt = integ.dt
    uprev = integ.uprev
    u = integ.u

    f!(u, uprev, p, t)
    @. u = uprev + dt * u

    integ.tprev = t
    integ.t += dt

    return nothing
end

@inline @muladd function DiffEqBase.step!(integ::SEI{false, S, T}) where {T, S}
    integ.uprev = integ.u
    f = integ.f
    p = integ.p
    t = integ.t
    dt = integ.dt
    uprev = integ.uprev

    k = f(uprev, p, t)
    integ.u = uprev + dt * k
    integ.tprev = t
    integ.t += dt

    return nothing
end

################################################################################
#                                Interpolation
################################################################################

@inline @muladd function (integ::SEI)(t::T) where {T}
    t₁, t₀, dt = integ.t, integ.tprev, integ.dt

    y₀ = integ.uprev
    y₁ = integ.u
    Θ = (t - t₀) / dt

    # Hermite interpolation.
    @inbounds u = (1 - Θ) * y₀ + Θ * y₁
    return u
end
