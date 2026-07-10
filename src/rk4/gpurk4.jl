#######################################################################################
# GPU-crutch solve method
# Makes the simplest possible method for GPU-compatibility
# Out of place only
#######################################################################################

"""
    GPUSimpleRK4()

Construct a GPU-compatible fixed-step fourth-order Runge-Kutta algorithm.

`GPUSimpleRK4` is an out-of-place RK4 implementation intended for GPU kernels
and static-array workloads.

# Arguments

No positional arguments are accepted.

# Keywords

No constructor keywords are accepted. Pass `dt` to `solve`.

# Returns

A `GPUSimpleRK4` algorithm object for use with out-of-place `ODEProblem`
definitions.

# Example

```julia
using SimpleDiffEq, StaticArrays

f(u, p, t) = -u

u0 = 1.0
tspan = (0.0, 10.0)
prob = ODEProblem(f, u0, tspan)

sol = solve(prob, GPUSimpleRK4(), dt = 0.1)
```

# Notes

`dt` is required. In-place problem functions are not supported.

# See Also

[`SimpleRK4`](@ref), [`LoopRK4`](@ref)
"""
struct GPUSimpleRK4 <: AbstractSimpleDiffEqODEAlgorithm end
export GPUSimpleRK4

@muladd function DiffEqBase.solve(
        prob::ODEProblem,
        alg::GPUSimpleRK4;
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
