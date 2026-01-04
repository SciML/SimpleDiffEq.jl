#######################################################################################
# GPU-crutch solve method
# Makes the simplest possible method for GPU-compatibility
# Out of place only
#######################################################################################

"""
    GPUSimpleVern7

GPU-compatible Verner 7th order Runge-Kutta method with fixed time step.

This is a 7th order explicit Runge-Kutta method designed for GPU compatibility.
It only supports out-of-place formulations and provides very high accuracy for
smooth problems.

## Example

```julia
using SimpleDiffEq

# Define ODE (out-of-place only)
f(u, p, t) = 1.01 * u

u0 = 0.5
tspan = (0.0, 10.0)
prob = ODEProblem(f, u0, tspan)

sol = solve(prob, GPUSimpleVern7(), dt = 0.1)
```

## Parameters

- `dt`: Fixed time step size (default: 0.1)
- `saveat`: Optional times to save solution at
- `save_everystep`: Save at every step (default: true)

## Restrictions

- Out-of-place formulations only
- Optimized for GPU execution

## See also

- [`GPUSimpleAVern7`](@ref) for the adaptive step size version
- [`GPUSimpleVern9`](@ref) for even higher order accuracy
"""
struct GPUSimpleVern7 <: AbstractSimpleDiffEqODEAlgorithm end
export GPUSimpleVern7

@muladd function DiffEqBase.solve(
        prob::ODEProblem,
        alg::GPUSimpleVern7; saveat = nothing,
        save_everystep = true,
        dt = 0.1f0
    )
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

    tab = Vern7Tableau(eltype(u0), eltype(u0))

    @unpack c2, c3, c4, c5, c6, c7, c8, a021, a031, a032, a041, a043, a051, a053, a054,
        a061, a063, a064, a065, a071, a073, a074, a075, a076, a081, a083, a084,
        a085, a086, a087, a091, a093, a094, a095, a096, a097, a098, a101, a103,
        a104, a105, a106, a107, b1, b4, b5, b6, b7, b8, b9 = tab

    @unpack c11, a1101, a1104, a1105, a1106, a1107, a1108, a1109, c12, a1201, a1204,
        a1205, a1206, a1207, a1208, a1209, a1211, c13, a1301, a1304, a1305, a1306, a1307,
        a1308, a1309, a1311, a1312, c14, a1401, a1404, a1405, a1406, a1407, a1408, a1409,
        a1411, a1412, a1413, c15, a1501, a1504, a1505, a1506, a1507, a1508, a1509, a1511,
        a1512, a1513, c16, a1601, a1604, a1605, a1606, a1607, a1608, a1609,
        a1611, a1612, a1613 = tab.extra

    _ts = tspan[1]:dt:tspan[2]

    # FSAL
    for i in 2:length(_ts)
        uprev = u
        t = _ts[i - 1]
        k1 = f(uprev, p, t)
        a = dt * a021
        k2 = f(uprev + a * k1, p, t + c2 * dt)
        k3 = f(uprev + dt * (a031 * k1 + a032 * k2), p, t + c3 * dt)
        k4 = f(uprev + dt * (a041 * k1 + a043 * k3), p, t + c4 * dt)
        k5 = f(uprev + dt * (a051 * k1 + a053 * k3 + a054 * k4), p, t + c5 * dt)
        k6 = f(uprev + dt * (a061 * k1 + a063 * k3 + a064 * k4 + a065 * k5), p, t + c6 * dt)
        k7 = f(
            uprev + dt * (a071 * k1 + a073 * k3 + a074 * k4 + a075 * k5 + a076 * k6), p,
            t + c7 * dt
        )
        k8 = f(
            uprev +
                dt * (a081 * k1 + a083 * k3 + a084 * k4 + a085 * k5 + a086 * k6 + a087 * k7),
            p,
            t + c8 * dt
        )
        g9 = uprev +
            dt *
            (
            a091 * k1 + a093 * k3 + a094 * k4 + a095 * k5 + a096 * k6 + a097 * k7 +
                a098 * k8
        )
        g10 = uprev +
            dt * (a101 * k1 + a103 * k3 + a104 * k4 + a105 * k5 + a106 * k6 + a107 * k7)
        k9 = f(g9, p, t + dt)
        k10 = f(g10, p, t + dt)

        u = uprev +
            dt * (b1 * k1 + b4 * k4 + b5 * k5 + b6 * k6 + b7 * k7 + b8 * k8 + b9 * k9)

        t += dt
        if saveat === nothing && save_everystep
            push!(us, u)
            push!(ts, t)
        elseif saveat !== nothing
            while cur_t <= length(ts) && ts[cur_t] <= t
                savet = ts[cur_t]
                θ = (savet - (t - dt)) / dt

                b1Θ, b4Θ, b5Θ, b6Θ, b7Θ, b8Θ, b9Θ, b11Θ, b12Θ,
                    b13Θ, b14Θ, b15Θ, b16Θ = bθs(
                    tab.interp,
                    θ
                )

                k11 = f(
                    uprev +
                        dt * (
                        a1101 * k1 + a1104 * k4 + a1105 * k5 + a1106 * k6 +
                            a1107 * k7 + a1108 * k8 + a1109 * k9
                    ),
                    p,
                    t + c11 * dt
                )
                k12 = f(
                    uprev +
                        dt * (
                        a1201 * k1 + a1204 * k4 + a1205 * k5 + a1206 * k6 +
                            a1207 * k7 + a1208 * k8 + a1209 * k9 + a1211 * k11
                    ),
                    p,
                    t + c12 * dt
                )
                k13 = f(
                    uprev +
                        dt * (
                        a1301 * k1 + a1304 * k4 + a1305 * k5 + a1306 * k6 +
                            a1307 * k7 + a1308 * k8 + a1309 * k9 + a1311 * k11 +
                            a1312 * k12
                    ),
                    p,
                    t + c13 * dt
                )
                k14 = f(
                    uprev +
                        dt * (
                        a1401 * k1 + a1404 * k4 + a1405 * k5 + a1406 * k6 +
                            a1407 * k7 + a1408 * k8 + a1409 * k9 + a1411 * k11 +
                            a1412 * k12 + a1413 * k13
                    ),
                    p,
                    t + c14 * dt
                )
                k15 = f(
                    uprev +
                        dt * (
                        a1501 * k1 + a1504 * k4 + a1505 * k5 + a1506 * k6 +
                            a1507 * k7 + a1508 * k8 + a1509 * k9 + a1511 * k11 +
                            a1512 * k12 + a1513 * k13
                    ),
                    p,
                    t + c15 * dt
                )
                k16 = f(
                    uprev +
                        dt * (
                        a1601 * k1 + a1604 * k4 + a1605 * k5 + a1606 * k6 +
                            a1607 * k7 + a1608 * k8 + a1609 * k9 + a1611 * k11 +
                            a1612 * k12 + a1613 * k13
                    ),
                    p,
                    t + c16 * dt
                )

                us[cur_t] = uprev +
                    dt * (
                    k1 * b1Θ
                        + k4 * b4Θ + k5 * b5Θ + k6 * b6Θ + k7 * b7Θ +
                        k8 * b8Θ + k9 * b9Θ
                        + k11 * b11Θ + k12 * b12Θ + k13 * b13Θ +
                        k14 * b14Θ + k15 * b15Θ + k16 * b16Θ
                )

                cur_t += 1
            end
        end
    end

    if saveat === nothing && !save_everystep
        push!(us, u)
        push!(ts, t)
    end

    sol = DiffEqBase.build_solution(
        prob, alg, ts, us,
        k = nothing, stats = nothing,
        calculate_error = false
    )
    DiffEqBase.has_analytic(prob.f) &&
        DiffEqBase.calculate_solution_errors!(
        sol; timeseries_errors = true,
        dense_errors = false
    )
    sol
end

#######################################################################################
# GPU-crutch solve method
# Makes the simplest possible adaptive method for GPU-compatibility
# Out of place only
#######################################################################################

"""
    GPUSimpleAVern7

GPU-compatible adaptive Verner 7th order Runge-Kutta method.

This is a GPU-optimized adaptive version of the Verner 7th order method with PI-controlled
adaptive stepping. It only supports out-of-place formulations and provides very high accuracy
with automatic step size control.

## Example

```julia
using SimpleDiffEq

# Define ODE (out-of-place only)
f(u, p, t) = 1.01 * u

u0 = 0.5
tspan = (0.0, 10.0)
prob = ODEProblem(f, u0, tspan)

sol = solve(prob, GPUSimpleAVern7(), dt = 0.1, abstol = 1e-6, reltol = 1e-3)
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

- [`GPUSimpleVern7`](@ref) for the fixed step size version
- [`GPUSimpleAVern9`](@ref) for even higher order adaptive method
- [`GPUSimpleATsit5`](@ref) for a lower order adaptive alternative
"""
struct GPUSimpleAVern7 end
export GPUSimpleAVern7

SciMLBase.isadaptive(alg::GPUSimpleAVern7) = true

@muladd function DiffEqBase.solve(
        prob::ODEProblem,
        alg::GPUSimpleAVern7;
        dt = 0.1f0, saveat = nothing,
        save_everystep = true,
        abstol = 1.0f-6, reltol = 1.0f-3
    )
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

    tab = Vern7Tableau(eltype(u0), eltype(u0))

    @unpack c2, c3, c4, c5, c6, c7, c8, a021, a031, a032, a041, a043, a051, a053, a054,
        a061, a063, a064, a065, a071, a073, a074, a075, a076, a081, a083, a084,
        a085, a086, a087, a091, a093, a094, a095, a096, a097, a098, a101, a103,
        a104, a105, a106, a107, b1, b4, b5, b6, b7, b8, b9, btilde1, btilde4,
        btilde5, btilde6, btilde7, btilde8, btilde9, btilde10, extra, interp = tab

    @unpack c11, a1101, a1104, a1105, a1106, a1107, a1108, a1109, c12, a1201, a1204,
        a1205, a1206, a1207, a1208, a1209, a1211, c13, a1301, a1304, a1305, a1306, a1307,
        a1308, a1309, a1311, a1312, c14, a1401, a1404, a1405, a1406, a1407, a1408, a1409,
        a1411, a1412, a1413, c15, a1501, a1504, a1505, a1506, a1507, a1508, a1509, a1511,
        a1512, a1513, c16, a1601, a1604, a1605, a1606, a1607, a1608, a1609,
        a1611, a1612, a1613 = tab.extra

    # FSAL
    while t < tspan[2]
        uprev = u
        EEst = Inf

        while EEst > 1
            dt < 1.0e-14 && error("dt<dtmin")

            k1 = f(uprev, p, t)
            a = dt * a021
            k2 = f(uprev + a * k1, p, t + c2 * dt)
            k3 = f(uprev + dt * (a031 * k1 + a032 * k2), p, t + c3 * dt)
            k4 = f(uprev + dt * (a041 * k1 + a043 * k3), p, t + c4 * dt)
            k5 = f(uprev + dt * (a051 * k1 + a053 * k3 + a054 * k4), p, t + c5 * dt)
            k6 = f(
                uprev + dt * (a061 * k1 + a063 * k3 + a064 * k4 + a065 * k5), p,
                t + c6 * dt
            )
            k7 = f(
                uprev + dt * (a071 * k1 + a073 * k3 + a074 * k4 + a075 * k5 + a076 * k6),
                p,
                t + c7 * dt
            )
            k8 = f(
                uprev +
                    dt *
                    (a081 * k1 + a083 * k3 + a084 * k4 + a085 * k5 + a086 * k6 + a087 * k7),
                p,
                t + c8 * dt
            )
            g9 = uprev +
                dt *
                (
                a091 * k1 + a093 * k3 + a094 * k4 + a095 * k5 + a096 * k6 + a097 * k7 +
                    a098 * k8
            )
            g10 = uprev +
                dt *
                (a101 * k1 + a103 * k3 + a104 * k4 + a105 * k5 + a106 * k6 + a107 * k7)
            k9 = f(g9, p, t + dt)
            k10 = f(g10, p, t + dt)

            u = uprev +
                dt * (b1 * k1 + b4 * k4 + b5 * k5 + b6 * k6 + b7 * k7 + b8 * k8 + b9 * k9)

            tmp = dt *
                (
                btilde1 * k1 + btilde4 * k4 + btilde5 * k5 + btilde6 * k6 +
                    btilde7 * k7 +
                    btilde8 * k8 + btilde9 * k9 + btilde10 * k10
            )
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
                if (tf - t - dtold) < 1.0e-14
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
                        b1Θ, b4Θ, b5Θ, b6Θ, b7Θ, b8Θ, b9Θ, b11Θ, b12Θ,
                            b13Θ, b14Θ, b15Θ, b16Θ = bθs(
                            tab.interp,
                            θ
                        )

                        k11 = f(
                            uprev +
                                dtold *
                                (
                                a1101 * k1 + a1104 * k4 + a1105 * k5 + a1106 * k6 +
                                    a1107 * k7 + a1108 * k8 + a1109 * k9
                            ),
                            p,
                            t + c11 * dtold
                        )
                        k12 = f(
                            uprev +
                                dtold *
                                (
                                a1201 * k1 + a1204 * k4 + a1205 * k5 + a1206 * k6 +
                                    a1207 * k7 + a1208 * k8 + a1209 * k9 + a1211 * k11
                            ),
                            p,
                            t + c12 * dtold
                        )
                        k13 = f(
                            uprev +
                                dtold *
                                (
                                a1301 * k1 + a1304 * k4 + a1305 * k5 + a1306 * k6 +
                                    a1307 * k7 + a1308 * k8 + a1309 * k9 + a1311 * k11 +
                                    a1312 * k12
                            ),
                            p,
                            t + c13 * dtold
                        )
                        k14 = f(
                            uprev +
                                dtold *
                                (
                                a1401 * k1 + a1404 * k4 + a1405 * k5 + a1406 * k6 +
                                    a1407 * k7 + a1408 * k8 + a1409 * k9 + a1411 * k11 +
                                    a1412 * k12 + a1413 * k13
                            ),
                            p,
                            t + c14 * dtold
                        )
                        k15 = f(
                            uprev +
                                dtold *
                                (
                                a1501 * k1 + a1504 * k4 + a1505 * k5 + a1506 * k6 +
                                    a1507 * k7 + a1508 * k8 + a1509 * k9 + a1511 * k11 +
                                    a1512 * k12 + a1513 * k13
                            ),
                            p,
                            t + c15 * dtold
                        )
                        k16 = f(
                            uprev +
                                dtold *
                                (
                                a1601 * k1 + a1604 * k4 + a1605 * k5 + a1606 * k6 +
                                    a1607 * k7 + a1608 * k8 + a1609 * k9 + a1611 * k11 +
                                    a1612 * k12 + a1613 * k13
                            ),
                            p,
                            t + c16 * dtold
                        )

                        us[cur_t] = uprev +
                            dtold * (
                            k1 * b1Θ
                                + k4 * b4Θ + k5 * b5Θ + k6 * b6Θ + k7 * b7Θ +
                                k8 * b8Θ + k9 * b9Θ
                                + k11 * b11Θ + k12 * b12Θ + k13 * b13Θ +
                                k14 * b14Θ + k15 * b15Θ + k16 * b16Θ
                        )

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
    sol = DiffEqBase.build_solution(
        prob, alg, ts, us,
        calculate_error = false
    )
    DiffEqBase.has_analytic(prob.f) &&
        DiffEqBase.calculate_solution_errors!(
        sol; timeseries_errors = true,
        dense_errors = false
    )
    sol
end
