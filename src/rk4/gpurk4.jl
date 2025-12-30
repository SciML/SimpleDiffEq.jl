#######################################################################################
# GPU-crutch solve method
# Makes the simplest possible method for GPU-compatibility
# Out of place only
#######################################################################################

"""
    GPUSimpleRK4

GPU-compatible classic Runge-Kutta 4th order method.

This is a simplified RK4 implementation designed for GPU compatibility. It only supports
out-of-place formulations and uses static arrays for efficiency on GPU architectures.

## Example

```julia
using SimpleDiffEq, StaticArrays

# Define ODE (out-of-place only)
f(u, p, t) = -u

u0 = 1.0
tspan = (0.0, 10.0)
prob = ODEProblem(f, u0, tspan)

sol = solve(prob, GPUSimpleRK4(), dt = 0.1)
```

## Required Parameters

- `dt`: Fixed time step size

## Restrictions

- Out-of-place formulations only (no in-place mutations)
- Optimized for GPU execution

## See also

- [`SimpleRK4`](@ref) for the standard CPU-optimized version
- [`LoopRK4`](@ref) for a teaching-focused variant
"""
struct GPUSimpleRK4 <: AbstractSimpleDiffEqODEAlgorithm end
export GPUSimpleRK4

@muladd function DiffEqBase.solve(prob::ODEProblem,
        alg::GPUSimpleRK4;
        dt = error("dt is required for this algorithm"))
    @assert !isinplace(prob)
    u0 = prob.u0
    tspan = prob.tspan
    f = prob.f
    p = prob.p
    t = tspan[1]
    tf = prob.tspan[2]
    ts = tspan[1]:dt:tspan[2]
    us = MVector{Int(length(ts)), typeof(u0)}(undef)
    us[1] = u0
    u = u0
    half = convert(eltype(u0), 1 // 2)
    sixth = convert(eltype(u0), 1 // 6)

    for i in 2:length(ts)
        uprev = u
        t = ts[i]
        k1 = f(u, p, t)
        tmp = uprev + dt * half * k1
        k2 = f(tmp, p, t + half * dt)
        tmp = uprev + dt * half * k2
        k3 = f(tmp, p, t + half * dt)
        tmp = uprev + dt * k3
        k4 = f(tmp, p, t + dt)
        u = uprev + dt * sixth * (k1 + 2k2 + 2k3 + k4)
        us[i] = u
    end

    sol = DiffEqBase.build_solution(prob, alg, ts, SArray(us),
        k = nothing, stats = nothing,
        calculate_error = false)
    DiffEqBase.has_analytic(prob.f) &&
        DiffEqBase.calculate_solution_errors!(sol; timeseries_errors = true,
            dense_errors = false)
    sol
end
