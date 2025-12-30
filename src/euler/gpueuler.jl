#######################################################################################
# GPU-crutch solve method
# Makes the simplest possible method for GPU-compatibility
# Out of place only
#######################################################################################

"""
    GPUSimpleEuler

GPU-compatible forward Euler method.

This is a simplified forward Euler implementation designed for GPU compatibility. It only
supports out-of-place formulations and uses static arrays for efficiency on GPU architectures.

## Example

```julia
using SimpleDiffEq

# Define ODE (out-of-place only)
f(u, p, t) = -0.5 * u

u0 = 1.0
tspan = (0.0, 10.0)
prob = ODEProblem(f, u0, tspan)

sol = solve(prob, GPUSimpleEuler(), dt = 0.1)
```

## Required Parameters

- `dt`: Fixed time step size

## Restrictions

- Out-of-place formulations only
- Optimized for GPU execution

## See also

- [`SimpleEuler`](@ref) for the standard CPU-optimized version
- [`GPUSimpleRK4`](@ref) for higher accuracy GPU solver
"""
struct GPUSimpleEuler <: AbstractSimpleDiffEqODEAlgorithm end
export GPUSimpleEuler

@muladd function DiffEqBase.solve(prob::ODEProblem,
        alg::GPUSimpleEuler;
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

    for i in 2:length(ts)
        uprev = u
        t = ts[i]
        k1 = f(u, p, t)
        u = uprev + dt * k1
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
