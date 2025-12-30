#######################################################################################
# Simplest Loop method
# Makes the simplest possible method for teaching and performance testing
#######################################################################################

"""
    LoopEuler

Simplified loop-based forward Euler method for teaching and benchmarking.

This is a minimal forward Euler implementation with explicit loops, designed for educational
purposes and to benchmark the overhead of more sophisticated integrators. It supports both
in-place and out-of-place formulations but without advanced features.

## Example

```julia
using SimpleDiffEq

# Define ODE
f(u, p, t) = -0.5 * u

u0 = 1.0
tspan = (0.0, 10.0)
prob = ODEProblem(f, u0, tspan)

sol = solve(prob, LoopEuler(), dt = 0.1)
```

## Required Parameters

- `dt`: Fixed time step size

## Use Cases

- Teaching how Euler's method works
- Benchmarking integrator overhead
- Minimal dependencies for simple problems

## See also

- [`SimpleEuler`](@ref) for a more optimized version
- [`LoopRK4`](@ref) for a higher order teaching variant
"""
struct LoopEuler <: AbstractSimpleDiffEqODEAlgorithm end
export LoopEuler

# Out-of-place
# No caching, good for static arrays, bad for arrays
@muladd function DiffEqBase.__solve(prob::ODEProblem{uType, tType, false},
    alg::LoopEuler;
    dt = error("dt is required for this algorithm"),
    save_everystep = true,
    save_start = true,
    adaptive = false,
    dense = false,
    save_end = true,
    kwargs...) where {uType, tType}
    @assert !adaptive
    @assert !dense
    u0 = prob.u0
    tspan = prob.tspan
    f = prob.f
    p = prob.p
    t = tspan[1]
    tf = prob.tspan[2]
    ts = tspan[1]:dt:tspan[2]

    if save_everystep && save_start
        us = Vector{typeof(u0)}(undef, length(ts))
        us[1] = u0
    elseif save_everystep
        us = Vector{typeof(u0)}(undef, length(ts) - 1)
    elseif save_start
        us = Vector{typeof(u0)}(undef, 2)
        us[1] = u0
    else
        us = Vector{typeof(u0)}(undef, 1) # for interface compatibility
    end

    u = u0

    for i in 2:length(ts)
        uprev = u
        t = ts[i]
        k = f(u, p, t)
        u = uprev + dt * k
        save_everystep && (us[i] = u)
    end

    !save_everystep && save_end && (us[end] = u)

    sol = DiffEqBase.build_solution(prob, alg, ts, us,
        k = nothing, stats = nothing,
        calculate_error = false)
    DiffEqBase.has_analytic(prob.f) &&
        DiffEqBase.calculate_solution_errors!(sol; timeseries_errors = true,
            dense_errors = false)
    sol
end

# In-place
# Good for mutable objects like arrays
# Use DiffEqBase.@.. for simd ivdep
@muladd function DiffEqBase.solve(prob::ODEProblem{uType, tType, true},
    alg::LoopEuler;
    dt = error("dt is required for this algorithm"),
    save_everystep = true,
    save_start = true,
    adaptive = false,
    dense = false,
    save_end = true,
    kwargs...) where {uType, tType}
    @assert !adaptive
    @assert !dense
    u0 = prob.u0
    tspan = prob.tspan
    f = prob.f
    p = prob.p
    t = tspan[1]
    tf = prob.tspan[2]
    ts = tspan[1]:dt:tspan[2]

    if save_everystep && save_start
        us = Vector{typeof(u0)}(undef, length(ts))
        us[1] = u0
    elseif save_everystep
        us = Vector{typeof(u0)}(undef, length(ts) - 1)
    elseif save_start
        us = Vector{typeof(u0)}(undef, 2)
        us[1] = u0
    else
        us = Vector{typeof(u0)}(undef, 1) # for interface compatibility
    end

    u = copy(u0)
    k = zero(u0)

    for i in 2:length(ts)
        t = ts[i]
        f(k, u, p, t)
        DiffEqBase.@.. u = u + dt * k
        save_everystep && (us[i] = copy(u))
    end

    !save_everystep && save_end && (us[end] = u)

    sol = DiffEqBase.build_solution(prob, alg, ts, us,
        k = nothing, stats = nothing,
        calculate_error = false)
    DiffEqBase.has_analytic(prob.f) &&
        DiffEqBase.calculate_solution_errors!(sol; timeseries_errors = true,
            dense_errors = false)
    sol
end
