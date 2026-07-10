"""
    SimpleFunctionMap()

Construct a simple function-map algorithm for `DiscreteProblem`s.

`SimpleFunctionMap` advances discrete dynamical systems with
`u[n+1] = f(u[n], p, t[n+1])` for out-of-place maps, or
`f!(u[n+1], u[n], p, t[n+1])` for in-place maps. The time step is fixed at one
unit of the problem time axis.

# Arguments

No positional arguments are accepted.

# Keywords

No constructor keywords are accepted. Pass solve-time options such as
`calculate_values` to `solve` or `init`.

# Returns

A `SimpleFunctionMap` algorithm object for use with `DiscreteProblem`.

# Example

```julia
using SimpleDiffEq

f(u, p, t) = p * u * (1 - u)

u0 = 0.5
tspan = (0.0, 100.0)
p = 3.2

prob = DiscreteProblem(f, u0, tspan, p)
sol = solve(prob, SimpleFunctionMap())
```

# Notes

Both in-place and out-of-place maps are supported. The algorithm also supports
the integrator interface through `init` and `step!`.

# See Also

`DiscreteProblem`
"""
struct SimpleFunctionMap end
export SimpleFunctionMap
SciMLBase.isdiscrete(alg::SimpleFunctionMap) = true

# ConstantCache version
function SciMLBase.__solve(
        prob::DiffEqBase.DiscreteProblem{uType, tupType, false},
        alg::SimpleFunctionMap;
        calculate_values = true, kwargs...
    ) where {uType, tupType}
    tType = eltype(tupType)
    tspan = prob.tspan
    f = prob.f
    u0 = prob.u0
    p = prob.p
    dt = 1

    n = Int((tspan[2] - tspan[1]) / dt + 1)
    u = [u0 for i in 1:n]
    t = tspan[1]:dt:tspan[2]
    if calculate_values
        for i in 2:n
            u[i] = f(u[i - 1], p, t[i])
        end
    end
    return sol = SciMLBase.build_solution(
        prob, alg, t, u, dense = false,
        interp = SciMLBase.ConstantInterpolation(t, u),
        calculate_error = false
    )
end

# Cache version
function SciMLBase.__solve(
        prob::DiscreteProblem{uType, tupType, true},
        alg::SimpleFunctionMap;
        calculate_values = true, kwargs...
    ) where {uType, tupType}
    tType = eltype(tupType)
    tspan = prob.tspan
    f = prob.f
    u0 = prob.u0
    p = prob.p
    dt = 1

    n = Int((tspan[2] - tspan[1]) / dt + 1)
    u = [similar(u0) for i in 1:n]
    u[1] .= u0
    t = tspan[1]:dt:tspan[2]
    if calculate_values
        for i in 2:n
            f(u[i], u[i - 1], p, t[i])
        end
    end
    return sol = SciMLBase.build_solution(
        prob, alg, t, u, dense = false,
        interp = SciMLBase.ConstantInterpolation(t, u),
        calculate_error = false
    )
end

##################################################

# Integrator version
mutable struct DiscreteIntegrator{F, IIP, uType, tType, P, S} <:
    SciMLBase.DEIntegrator{SimpleFunctionMap, IIP, uType, tType}
    f::F
    u::uType
    t::tType
    uprev::uType
    p::P
    sol::S
    i::Int
    tdir::tType
end

function SciMLBase.__init(
        prob::DiscreteProblem,
        alg::SimpleFunctionMap;
        kwargs...
    )
    sol = SciMLBase.__solve(prob, alg; calculate_values = false)
    F = typeof(prob.f)
    IIP = isinplace(prob)
    uType = typeof(prob.u0)
    tType = typeof(prob.tspan[1])
    P = typeof(prob.p)
    S = typeof(sol)
    return DiscreteIntegrator{F, IIP, uType, tType, P, S}(
        prob.f, prob.u0, prob.tspan[1],
        copy(prob.u0), prob.p, sol, 1,
        one(tType)
    )
end

function DiffEqBase.step!(integrator::DiscreteIntegrator)
    integrator.t = integrator.i
    integrator.i += 1
    u = integrator.u
    uprev = integrator.uprev
    p = integrator.p
    f = integrator.f
    i = integrator.i

    return if isinplace(integrator.sol.prob)
        f(integrator.sol.u[i], uprev, p, i)
        integrator.uprev = integrator.u
        integrator.u = integrator.sol.u[i]
    else
        u = f(uprev, p, i)
        integrator.sol.u[i] = u
        integrator.uprev = integrator.u
        integrator.u = u
    end
end
