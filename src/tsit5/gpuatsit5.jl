#######################################################################################
# GPU-crutch solve method
# Makes the simplest possible method for GPU-compatibility
# Out of place only
#######################################################################################

"""
    GPUSimpleTsit5

GPU-compatible Tsitouras 5th order Runge-Kutta method with fixed time step.

This is a GPU-optimized version of the Tsitouras 5th order method with a fixed step size.
It only supports out-of-place formulations for compatibility with GPU kernels.

## Example

```julia
using SimpleDiffEq

# Define ODE (out-of-place only)
f(u, p, t) = 1.01 * u

u0 = 0.5
tspan = (0.0, 10.0)
prob = ODEProblem(f, u0, tspan)

sol = solve(prob, GPUSimpleTsit5(), dt = 0.1)
```

## Parameters

- `dt`: Fixed time step size (default: 0.1)
- `saveat`: Optional times to save solution at
- `save_everystep`: Save at every step (default: true)

## Restrictions

- Out-of-place formulations only
- Optimized for GPU execution

## See also

- [`GPUSimpleATsit5`](@ref) for the adaptive step size version
- [`SimpleTsit5`](@ref) for the CPU-optimized version
"""
struct GPUSimpleTsit5 <: AbstractSimpleDiffEqODEAlgorithm end
export GPUSimpleTsit5

@muladd function DiffEqBase.solve(prob::ODEProblem,
        alg::GPUSimpleTsit5; saveat = nothing,
        save_everystep = true,
        dt = 0.1f0)
    @assert !isinplace(prob)
    u0 = prob.u0
    tspan = prob.tspan
    f = prob.f
    p = prob.p
    t = tspan[1]
    tf = tspan[2]

    if saveat === nothing
        ts = Vector{eltype(dt)}(undef, 1)
        ts[1] = prob.tspan[1]
        us = Vector{typeof(u0)}(undef, 0)
        push!(us, recursivecopy(u0))
    else
        ts = saveat
        cur_t = 1
        us = MVector{Int(length(ts)), typeof(u0)}(undef)
        if prob.tspan[1] == ts[1]
            cur_t += 1
            us[1] = u0
        end
    end

    u = u0
    k7 = f(u, p, t)

    cs, as, btildes, rs = _build_atsit5_caches(eltype(u0))
    c1, c2, c3, c4, c5, c6 = cs
    a21, a31, a32, a41, a42, a43, a51, a52, a53, a54,
    a61, a62, a63, a64, a65, a71, a72, a73, a74, a75, a76 = as
    _ts = tspan[1]:dt:tspan[2]

    # FSAL
    for i in 2:length(_ts)
        uprev = u
        k1 = k7
        t = _ts[i - 1]
        tmp = uprev + dt * a21 * k1
        k2 = f(tmp, p, t + c1 * dt)
        tmp = uprev + dt * (a31 * k1 + a32 * k2)
        k3 = f(tmp, p, t + c2 * dt)
        tmp = uprev + dt * (a41 * k1 + a42 * k2 + a43 * k3)
        k4 = f(tmp, p, t + c3 * dt)
        tmp = uprev + dt * (a51 * k1 + a52 * k2 + a53 * k3 + a54 * k4)
        k5 = f(tmp, p, t + c4 * dt)
        tmp = uprev + dt * (a61 * k1 + a62 * k2 + a63 * k3 + a64 * k4 + a65 * k5)
        k6 = f(tmp, p, t + dt)
        u = uprev + dt * (a71 * k1 + a72 * k2 + a73 * k3 + a74 * k4 + a75 * k5 + a76 * k6)
        k7 = f(u, p, t + dt)
        t += dt
        if saveat === nothing && save_everystep
            push!(us, u)
            push!(ts, t)
        elseif saveat !== nothing
            while cur_t <= length(ts) && ts[cur_t] <= t
                savet = ts[cur_t]
                θ = (savet - (t - dt)) / dt
                b1θ, b2θ, b3θ, b4θ, b5θ, b6θ, b7θ = bθs(rs, θ)
                us[cur_t] = uprev +
                            dt *
                            (b1θ * k1 + b2θ * k2 + b3θ * k3 + b4θ * k4 + b5θ * k5 +
                             b6θ * k6 + b7θ * k7)
                cur_t += 1
            end
        end
    end

    if saveat === nothing && !save_everystep
        push!(us, u)
        push!(ts, t)
    end

    sol = DiffEqBase.build_solution(prob, alg, ts, us,
        k = nothing, stats = nothing,
        calculate_error = false)
    DiffEqBase.has_analytic(prob.f) &&
        DiffEqBase.calculate_solution_errors!(sol; timeseries_errors = true,
            dense_errors = false)
    sol
end

#######################################################################################
# GPU-crutch solve method
# Makes the simplest possible adaptive method for GPU-compatibility
# Out of place only
#######################################################################################

"""
    GPUSimpleATsit5

GPU-compatible adaptive Tsitouras 5th order Runge-Kutta method.

This is a GPU-optimized version of the adaptive Tsitouras 5th order method with
PI-controlled adaptive stepping. It only supports out-of-place formulations.

## Example

```julia
using SimpleDiffEq

# Define ODE (out-of-place only)
f(u, p, t) = 1.01 * u

u0 = 0.5
tspan = (0.0, 10.0)
prob = ODEProblem(f, u0, tspan)

sol = solve(prob, GPUSimpleATsit5(), dt = 0.1, abstol = 1e-6, reltol = 1e-3)
```

## Parameters

- `dt`: Initial time step size (default: 0.1)
- `abstol`: Absolute tolerance (default: 1e-6)
- `reltol`: Relative tolerance (default: 1e-3)
- `saveat`: Optional times to save solution at
- `save_everystep`: Save at every step (default: true)

## Restrictions

- Out-of-place formulations only
- Optimized for GPU execution

## See also

- [`GPUSimpleTsit5`](@ref) for the fixed step size version
- [`SimpleATsit5`](@ref) for the CPU-optimized version
- [`GPUSimpleAVern7`](@ref) for higher order adaptive method
"""
struct GPUSimpleATsit5 end
export GPUSimpleATsit5

SciMLBase.isadaptive(alg::GPUSimpleATsit5) = true

@muladd function DiffEqBase.solve(prob::ODEProblem,
        alg::GPUSimpleATsit5;
        dt = 0.1f0, saveat = nothing,
        save_everystep = true,
        abstol = 1.0f-6, reltol = 1.0f-3)
    @assert !isinplace(prob)
    u0 = prob.u0
    tspan = prob.tspan
    f = prob.f
    p = prob.p
    beta1, beta2, qmax, qmin, gamma, qoldinit,
    _ = build_adaptive_controller_cache(eltype(u0))

    t = tspan[1]
    tf = prob.tspan[2]

    if saveat === nothing
        ts = Vector{eltype(dt)}(undef, 1)
        ts[1] = prob.tspan[1]
        us = Vector{typeof(u0)}(undef, 0)
        push!(us, recursivecopy(u0))
    else
        ts = saveat
        cur_t = 1
        us = MVector{Int(length(ts)), typeof(u0)}(undef)
        if prob.tspan[1] == ts[1]
            cur_t += 1
            us[1] = u0
        end
    end

    u = u0
    qold = qoldinit
    k7 = f(u, p, t)

    cs, as, btildes, rs = _build_atsit5_caches(eltype(u0))
    c1, c2, c3, c4, c5, c6 = cs
    a21, a31, a32, a41, a42, a43, a51, a52, a53, a54,
    a61, a62, a63, a64, a65, a71, a72, a73, a74, a75, a76 = as
    btilde1, btilde2, btilde3, btilde4, btilde5, btilde6, btilde7 = btildes

    # FSAL
    while t < tspan[2]
        uprev = u
        k1 = k7
        EEst = Inf

        while EEst > 1
            dt < 1e-14 && error("dt<dtmin")

            tmp = uprev + dt * a21 * k1
            k2 = f(tmp, p, t + c1 * dt)
            tmp = uprev + dt * (a31 * k1 + a32 * k2)
            k3 = f(tmp, p, t + c2 * dt)
            tmp = uprev + dt * (a41 * k1 + a42 * k2 + a43 * k3)
            k4 = f(tmp, p, t + c3 * dt)
            tmp = uprev + dt * (a51 * k1 + a52 * k2 + a53 * k3 + a54 * k4)
            k5 = f(tmp, p, t + c4 * dt)
            tmp = uprev + dt * (a61 * k1 + a62 * k2 + a63 * k3 + a64 * k4 + a65 * k5)
            k6 = f(tmp, p, t + dt)
            u = uprev +
                dt * (a71 * k1 + a72 * k2 + a73 * k3 + a74 * k4 + a75 * k5 + a76 * k6)
            k7 = f(u, p, t + dt)

            tmp = dt * (btilde1 * k1 + btilde2 * k2 + btilde3 * k3 + btilde4 * k4 +
                   btilde5 * k5 + btilde6 * k6 + btilde7 * k7)
            tmp = tmp ./ (abstol .+ max.(abs.(uprev), abs.(u)) * reltol)
            EEst = DiffEqBase.ODE_DEFAULT_NORM(tmp, t)

            if iszero(EEst)
                q = inv(qmax)
            else
                @fastmath q11 = EEst^beta1
                @fastmath q = q11 / (qold^beta2)
            end

            if EEst > 1
                dt = dt / min(inv(qmin), q11 / gamma)
            else # EEst <= 1
                @fastmath q = max(inv(qmax), min(inv(qmin), q / gamma))
                qold = max(EEst, qoldinit)
                dtold = dt
                dt = dt / q #dtnew
                dt = min(abs(dt), abs(tf - t - dtold))
                told = t
                if (tf - t - dtold) < 1e-14
                    t = tf
                else
                    t += dtold
                end

                if saveat === nothing && save_everystep
                    push!(us, recursivecopy(u))
                    push!(ts, t)
                elseif saveat !== nothing
                    while cur_t <= length(ts) && ts[cur_t] <= t
                        savet = ts[cur_t]
                        θ = (savet - told) / dtold
                        b1θ, b2θ, b3θ, b4θ, b5θ, b6θ, b7θ = bθs(rs, θ)
                        us[cur_t] = uprev +
                                    dtold *
                                    (b1θ * k1 + b2θ * k2 + b3θ * k3 + b4θ * k4 + b5θ * k5 +
                                     b6θ * k6 + b7θ * k7)
                        cur_t += 1
                    end
                end
            end
        end
    end

    if saveat === nothing && !save_everystep
        push!(us, u)
        push!(ts, t)
    end
    sol = DiffEqBase.build_solution(prob, alg, ts, us,
        calculate_error = false)
    DiffEqBase.has_analytic(prob.f) &&
        DiffEqBase.calculate_solution_errors!(sol; timeseries_errors = true,
            dense_errors = false)
    sol
end
