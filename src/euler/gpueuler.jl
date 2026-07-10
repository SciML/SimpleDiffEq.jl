#######################################################################################
# GPU-crutch solve method
# Makes the simplest possible method for GPU-compatibility
# Out of place only
#######################################################################################

"""
    GPUSimpleEuler()

Construct a GPU-compatible fixed-step forward Euler algorithm.

`GPUSimpleEuler` is a small out-of-place Euler implementation intended for GPU
kernels and static-array workloads.

# Arguments

No positional arguments are accepted.

# Keywords

No constructor keywords are accepted. Pass `dt` to `solve`.

# Returns

A `GPUSimpleEuler` algorithm object for use with out-of-place `ODEProblem`
definitions.

# Example

```julia
using SimpleDiffEq

f(u, p, t) = -0.5 * u

u0 = 1.0
tspan = (0.0, 10.0)
prob = ODEProblem(f, u0, tspan)

sol = solve(prob, GPUSimpleEuler(), dt = 0.1)
```

# Notes

`dt` is required. In-place problem functions are not supported.

# See Also

[`SimpleEuler`](@ref), [`GPUSimpleRK4`](@ref)
"""
struct GPUSimpleEuler <: AbstractSimpleDiffEqODEAlgorithm end
export GPUSimpleEuler

@muladd function DiffEqBase.solve(
        prob::ODEProblem,
        alg::GPUSimpleEuler;
        dt = error("dt is required for this algorithm"),
        kwargs...
    )
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

    sol = SciMLBase.build_solution(
        prob, alg, ts, SArray(us),
        k = nothing, stats = nothing,
        calculate_error = false
    )
    SciMLBase.has_analytic(prob.f) &&
        SciMLBase.calculate_solution_errors!(
        sol; timeseries_errors = true,
        dense_errors = false
    )
    sol
end
