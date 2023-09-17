# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
# Euler solver
#
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

struct SimpleEuler <: AbstractSimpleDiffEqODEAlgorithm end
export SimpleEuler

mutable struct SimpleEulerIntegrator{IIP, S, T, P, F} <:
               DiffEqBase.AbstractODEIntegrator{SimpleEuler, IIP, S, T}
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

function DiffEqBase.__init(prob::ODEProblem, alg::SimpleEuler;
    dt = error("dt is required for this algorithm"))
    simpleeuler_init(prob.f,
        DiffEqBase.isinplace(prob),
        prob.u0,
        prob.tspan[1],
        dt,
        prob.p)
end

function DiffEqBase.__solve(prob::ODEProblem, alg::SimpleEuler;
    dt = error("dt is required for this algorithm"))
    u0 = prob.u0
    tspan = prob.tspan
    ts = Array(tspan[1]:dt:tspan[2])
    n = length(ts)
    us = Vector{typeof(u0)}(undef, n)

    @inbounds us[1] = _copy(u0)

    integ = simpleeuler_init(prob.f, DiffEqBase.isinplace(prob), prob.u0,
        prob.tspan[1], dt, prob.p)

    for i in 1:(n - 1)
        step!(integ)
        us[i + 1] = _copy(integ.u)
    end

    sol = DiffEqBase.build_solution(prob, alg, ts, us, calculate_error = false)

    DiffEqBase.has_analytic(prob.f) &&
        DiffEqBase.calculate_solution_errors!(sol;
            timeseries_errors = true,
            dense_errors = false)

    return sol
end

@inline function simpleeuler_init(f::F, IIP::Bool, u0::S, t0::T, dt::T,
    p::P) where
    {F, P, T, S}
    integ = SEI{IIP, S, T, P, F}(f,
        _copy(u0),
        _copy(u0),
        _copy(u0),
        t0,
        t0,
        t0,
        dt,
        sign(dt),
        p,
        true)

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
    @inbounds if !isinplace(integ)
        u = (1 - Θ) * y₀ + Θ * y₁
        return u
    else
        for i in 1:length(u)
            u = @. (1 - Θ) * y₀ + Θ * y₁
        end
        return u
    end
end
