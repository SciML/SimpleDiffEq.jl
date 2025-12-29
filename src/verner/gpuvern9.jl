#######################################################################################
# GPU-crutch solve method
# Makes the simplest possible method for GPU-compatibility
# Out of place only
#######################################################################################
struct GPUSimpleVern9 <: AbstractSimpleDiffEqODEAlgorithm end
export GPUSimpleVern9

@muladd function DiffEqBase.solve(prob::ODEProblem,
        alg::GPUSimpleVern9; saveat = nothing,
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

    tab = Vern9Tableau(eltype(u0), eltype(u0))

    @unpack c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, a0201, a0301, a0302,
    a0401, a0403, a0501, a0503, a0504, a0601, a0604, a0605, a0701, a0704, a0705, a0706,
    a0801, a0806, a0807, a0901, a0906, a0907, a0908, a1001, a1006, a1007, a1008, a1009,
    a1101, a1106, a1107, a1108, a1109, a1110, a1201, a1206, a1207, a1208, a1209, a1210,
    a1211, a1301, a1306, a1307, a1308, a1309, a1310, a1311, a1312, a1401, a1406, a1407,
    a1408, a1409, a1410, a1411, a1412, a1413, a1501, a1506, a1507, a1508, a1509, a1510,
    a1511, a1512, a1513, a1514, a1601, a1606, a1607, a1608, a1609, a1610, a1611, a1612,
    a1613, b1, b8, b9, b10, b11, b12, b13, b14, b15, btilde1, btilde8, btilde9, btilde10,
    btilde11, btilde12, btilde13, btilde14, btilde15, btilde16 = tab

    _ts = tspan[1]:dt:tspan[2]

    # FSAL
    for i in 2:length(_ts)
        uprev = u
        t = _ts[i - 1]
        k1 = f(uprev, p, t)
        a = dt * a0201
        k2 = f(uprev + a * k1, p, t + c1 * dt)
        k3 = f(uprev + dt * (a0301 * k1 + a0302 * k2), p, t + c2 * dt)
        k4 = f(uprev + dt * (a0401 * k1 + a0403 * k3), p, t + c3 * dt)
        k5 = f(uprev + dt * (a0501 * k1 + a0503 * k3 + a0504 * k4), p, t + c4 * dt)
        k6 = f(uprev + dt * (a0601 * k1 + a0604 * k4 + a0605 * k5), p, t + c5 * dt)
        k7 = f(uprev + dt * (a0701 * k1 + a0704 * k4 + a0705 * k5 + a0706 * k6), p,
            t + c6 * dt)
        k8 = f(uprev + dt * (a0801 * k1 + a0806 * k6 + a0807 * k7), p, t + c7 * dt)
        k9 = f(uprev + dt * (a0901 * k1 + a0906 * k6 + a0907 * k7 + a0908 * k8), p,
            t + c8 * dt)
        k10 = f(
            uprev +
            dt * (a1001 * k1 + a1006 * k6 + a1007 * k7 + a1008 * k8 + a1009 * k9),
            p, t + c9 * dt)
        k11 = f(
            uprev +
            dt *
            (a1101 * k1 + a1106 * k6 + a1107 * k7 + a1108 * k8 + a1109 * k9 +
             a1110 * k10),
            p, t + c10 * dt)
        k12 = f(
            uprev +
            dt *
            (a1201 * k1 + a1206 * k6 + a1207 * k7 + a1208 * k8 + a1209 * k9 +
             a1210 * k10 +
             a1211 * k11),
            p,
            t + c11 * dt)
        k13 = f(
            uprev +
            dt *
            (a1301 * k1 + a1306 * k6 + a1307 * k7 + a1308 * k8 + a1309 * k9 +
             a1310 * k10 +
             a1311 * k11 + a1312 * k12),
            p,
            t + c12 * dt)
        k14 = f(
            uprev +
            dt *
            (a1401 * k1 + a1406 * k6 + a1407 * k7 + a1408 * k8 + a1409 * k9 +
             a1410 * k10 +
             a1411 * k11 + a1412 * k12 + a1413 * k13),
            p,
            t + c13 * dt)
        g15 = uprev +
              dt *
              (a1501 * k1 + a1506 * k6 + a1507 * k7 + a1508 * k8 + a1509 * k9 +
               a1510 * k10 +
               a1511 * k11 + a1512 * k12 + a1513 * k13 + a1514 * k14)

        k15 = f(g15, p, t + dt)

        u = uprev +
            dt *
            (b1 * k1 + b8 * k8 + b9 * k9 + b10 * k10 + b11 * k11 + b12 * k12 + b13 * k13 +
             b14 * k14 + b15 * k15)

        t += dt
        if saveat === nothing && save_everystep
            push!(us, u)
            push!(ts, t)
        elseif saveat !== nothing
            while cur_t <= length(ts) && ts[cur_t] <= t
                savet = ts[cur_t]
                θ = (savet - (t - dt)) / dt
                b1Θ, b8Θ, b9Θ, b10Θ, b11Θ, b12Θ, b13Θ, b14Θ, b15Θ, b17Θ, b18Θ, b19Θ, b20Θ,
                b21Θ, b22Θ, b23Θ, b24Θ, b25Θ, b26Θ = bθs(tab.interp, θ)

                @unpack c17,
                a1701, a1708, a1709, a1710, a1711, a1712, a1713, a1714, a1715, c18, a1801,
                a1808, a1809, a1810, a1811, a1812, a1813, a1814,
                a1815, a1817, c19, a1901, a1908, a1909,
                a1910, a1911, a1912, a1913, a1914, a1915, a1917,
                a1918, c20, a2001, a2008, a2009, a2010,
                a2011, a2012, a2013, a2014, a2015, a2017, a2018,
                a2019, c21, a2101, a2108, a2109, a2110,
                a2111, a2112, a2113, a2114, a2115, a2117, a2118,
                a2119, a2120, c22, a2201, a2208, a2209,
                a2210, a2211, a2212, a2213, a2214, a2215, a2217,
                a2218, a2219, a2220, a2221, c23, a2301,
                a2308, a2309, a2310, a2311, a2312, a2313,
                a2314, a2315, a2317, a2318, a2319, a2320,
                a2321, c24, a2401, a2408, a2409, a2410, a2411, a2412, a2413, a2414, a2415, a2417,
                a2418, a2419, a2420, a2421, c25, a2501, a2508, a2509, a2510, a2511, a2512, a2513,
                a2514, a2515, a2517, a2518, a2519, a2520, a2521, c26, a2601, a2608, a2609, a2610,
                a2611, a2612, a2613, a2614, a2615, a2617, a2618, a2619, a2620,
                a2621 = tab.extra

                k11 = f(
                    uprev +
                    dt * (a1701 * k1 + a1708 * k2 + a1709 * k3 + a1710 * k4 +
                     a1711 * k5 + a1712 * k6 + a1713 * k7 + a1714 * k8 + a1715 * k9),
                    p, t + c17 * dt)
                k12 = f(
                    uprev +
                    dt * (a1801 * k1 + a1808 * k2 + a1809 * k3 + a1810 * k4 +
                     a1811 * k5 + a1812 * k6 + a1813 * k7 + a1814 * k8 +
                     a1815 * k9 + a1817 * k11),
                    p,
                    t + c18 * dt)
                k13 = f(
                    uprev +
                    dt * (a1901 * k1 + a1908 * k2 + a1909 * k3 + a1910 * k4 +
                     a1911 * k5 + a1912 * k6 + a1913 * k7 + a1914 * k8 +
                     a1915 * k9 + a1917 * k11 + a1918 * k12),
                    p,
                    t + c19 * dt)
                k14 = f(
                    uprev +
                    dt * (a2001 * k1 + a2008 * k2 + a2009 * k3 + a2010 * k4 +
                     a2011 * k5 + a2012 * k6 + a2013 * k7 + a2014 * k8 +
                     a2015 * k9 + a2017 * k11 + a2018 * k12 + a2019 * k13),
                    p,
                    t + c20 * dt)
                k15 = f(
                    uprev +
                    dt * (a2101 * k1 + a2108 * k2 + a2109 * k3 + a2110 * k4 +
                     a2111 * k5 + a2112 * k6 + a2113 * k7 + a2114 * k8 +
                     a2115 * k9 + a2117 * k11 + a2118 * k12 + a2119 * k13 +
                     a2120 * k14),
                    p,
                    t + c21 * dt)
                k16 = f(
                    uprev +
                    dt * (a2201 * k1 + a2208 * k2 + a2209 * k3 + a2210 * k4 +
                     a2211 * k5 + a2212 * k6 + a2213 * k7 + a2214 * k8 +
                     a2215 * k9 + a2217 * k11 + a2218 * k12 + a2219 * k13 +
                     a2220 * k14 + a2221 * k15),
                    p,
                    t + c22 * dt)
                k17 = f(
                    uprev +
                    dt * (a2301 * k1 + a2308 * k2 + a2309 * k3 + a2310 * k4 +
                     a2311 * k5 + a2312 * k6 + a2313 * k7 + a2314 * k8 +
                     a2315 * k9 + a2317 * k11 + a2318 * k12 + a2319 * k13 +
                     a2320 * k14 + a2321 * k15),
                    p,
                    t + c23 * dt)
                k18 = f(
                    uprev +
                    dt * (a2401 * k1 + a2408 * k2 + a2409 * k3 + a2410 * k4 +
                     a2411 * k5 + a2412 * k6 + a2413 * k7 + a2414 * k8 +
                     a2415 * k9 + a2417 * k11 + a2418 * k12 + a2419 * k13 +
                     a2420 * k14 + a2421 * k15),
                    p,
                    t + c24 * dt)
                k19 = f(
                    uprev +
                    dt * (a2501 * k1 + a2508 * k2 + a2509 * k3 + a2510 * k4 +
                     a2511 * k5 + a2512 * k6 + a2513 * k7 + a2514 * k8 +
                     a2515 * k9 + a2517 * k11 + a2518 * k12 + a2519 * k13 +
                     a2520 * k14 + a2521 * k15),
                    p,
                    t + c25 * dt)
                k20 = f(
                    uprev +
                    dt * (a2601 * k1 + a2608 * k2 + a2609 * k3 + a2610 * k4 +
                     a2611 * k5 + a2612 * k6 + a2613 * k7 + a2614 * k8 +
                     a2615 * k9 + a2617 * k11 + a2618 * k12 + a2619 * k13 +
                     a2620 * k14 + a2621 * k15),
                    p,
                    t + c26 * dt)

                us[cur_t] = uprev +
                            dt *
                            (k1 * b1Θ + k2 * b8Θ + k3 * b9Θ + k4 * b10Θ +
                             k5 * b11Θ +
                             k6 * b12Θ + k7 * b13Θ + k8 * b14Θ + k9 * b15Θ +
                             k11 * b17Θ +
                             k12 * b18Θ + k13 * b19Θ + k14 * b20Θ + k15 * b21Θ +
                             k16 * b22Θ +
                             k17 * b23Θ + k18 * b24Θ + k19 * b25Θ + k20 * b26Θ)
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
struct GPUSimpleAVern9 end
export GPUSimpleAVern9

SciMLBase.isadaptive(alg::GPUSimpleAVern9) = true

@muladd function DiffEqBase.solve(prob::ODEProblem,
        alg::GPUSimpleAVern9;
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

    tab = Vern9Tableau(eltype(u0), eltype(u0))

    @unpack c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, a0201, a0301, a0302,
    a0401, a0403, a0501, a0503, a0504, a0601, a0604, a0605, a0701, a0704, a0705, a0706,
    a0801, a0806, a0807, a0901, a0906, a0907, a0908, a1001, a1006, a1007, a1008, a1009,
    a1101, a1106, a1107, a1108, a1109, a1110, a1201, a1206, a1207, a1208, a1209, a1210,
    a1211, a1301, a1306, a1307, a1308, a1309, a1310, a1311, a1312, a1401, a1406, a1407,
    a1408, a1409, a1410, a1411, a1412, a1413, a1501, a1506, a1507, a1508, a1509, a1510,
    a1511, a1512, a1513, a1514, a1601, a1606, a1607, a1608, a1609, a1610, a1611, a1612,
    a1613, b1, b8, b9, b10, b11, b12, b13, b14, b15, btilde1, btilde8, btilde9, btilde10,
    btilde11, btilde12, btilde13, btilde14, btilde15, btilde16 = tab

    # FSAL
    while t < tspan[2]
        uprev = u
        EEst = Inf

        while EEst > 1
            dt < 1.0f-7 && error("dt<dtmin")

            k1 = f(uprev, p, t)
            a = dt * a0201
            k2 = f(uprev + a * k1, p, t + c1 * dt)
            k3 = f(uprev + dt * (a0301 * k1 + a0302 * k2), p, t + c2 * dt)
            k4 = f(uprev + dt * (a0401 * k1 + a0403 * k3), p, t + c3 * dt)
            k5 = f(uprev + dt * (a0501 * k1 + a0503 * k3 + a0504 * k4), p, t + c4 * dt)
            k6 = f(uprev + dt * (a0601 * k1 + a0604 * k4 + a0605 * k5), p, t + c5 * dt)
            k7 = f(uprev + dt * (a0701 * k1 + a0704 * k4 + a0705 * k5 + a0706 * k6), p,
                t + c6 * dt)
            k8 = f(uprev + dt * (a0801 * k1 + a0806 * k6 + a0807 * k7), p, t + c7 * dt)
            k9 = f(uprev + dt * (a0901 * k1 + a0906 * k6 + a0907 * k7 + a0908 * k8), p,
                t + c8 * dt)
            k10 = f(
                uprev +
                dt * (a1001 * k1 + a1006 * k6 + a1007 * k7 + a1008 * k8 + a1009 * k9),
                p, t + c9 * dt)
            k11 = f(
                uprev +
                dt *
                (a1101 * k1 + a1106 * k6 + a1107 * k7 + a1108 * k8 + a1109 * k9 +
                 a1110 * k10),
                p, t + c10 * dt)
            k12 = f(
                uprev +
                dt *
                (a1201 * k1 + a1206 * k6 + a1207 * k7 + a1208 * k8 + a1209 * k9 +
                 a1210 * k10 +
                 a1211 * k11),
                p,
                t + c11 * dt)
            k13 = f(
                uprev +
                dt *
                (a1301 * k1 + a1306 * k6 + a1307 * k7 + a1308 * k8 + a1309 * k9 +
                 a1310 * k10 +
                 a1311 * k11 + a1312 * k12),
                p,
                t + c12 * dt)
            k14 = f(
                uprev +
                dt *
                (a1401 * k1 + a1406 * k6 + a1407 * k7 + a1408 * k8 + a1409 * k9 +
                 a1410 * k10 +
                 a1411 * k11 + a1412 * k12 + a1413 * k13),
                p,
                t + c13 * dt)
            g15 = uprev +
                  dt *
                  (a1501 * k1 + a1506 * k6 + a1507 * k7 + a1508 * k8 + a1509 * k9 +
                   a1510 * k10 +
                   a1511 * k11 + a1512 * k12 + a1513 * k13 + a1514 * k14)
            g16 = uprev +
                  dt *
                  (a1601 * k1 + a1606 * k6 + a1607 * k7 + a1608 * k8 + a1609 * k9 +
                   a1610 * k10 +
                   a1611 * k11 + a1612 * k12 + a1613 * k13)
            k15 = f(g15, p, t + dt)
            k16 = f(g16, p, t + dt)

            u = uprev +
                dt *
                (b1 * k1 + b8 * k8 + b9 * k9 + b10 * k10 + b11 * k11 + b12 * k12 +
                 b13 * k13 +
                 b14 * k14 + b15 * k15)

            tmp = dt * (btilde1 * k1 + btilde8 * k8 + btilde9 * k9 + btilde10 * k10 +
                   btilde11 * k11 + btilde12 * k12 + btilde13 * k13 + btilde14 * k14 +
                   btilde15 * k15 + btilde16 * k16)
            tmp = tmp ./ (abstol .+ max.(abs.(uprev), abs.(u)) * reltol)
            EEst = DiffEqBase.ODE_DEFAULT_NORM(tmp, t)

            k1 = k1
            k2 = k8
            k3 = k9
            k4 = k10
            k5 = k11
            k6 = k12
            k7 = k13
            k8 = k14
            k9 = k15
            k10 = k16

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
                if (tf - t - dtold) < 1.0f-7
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
                        b1Θ, b8Θ, b9Θ, b10Θ, b11Θ, b12Θ, b13Θ,
                        b14Θ, b15Θ, b17Θ, b18Θ, b19Θ, b20Θ,
                        b21Θ, b22Θ, b23Θ, b24Θ, b25Θ, b26Θ = bθs(tab.interp, θ)

                        @unpack c17, a1701, a1708, a1709, a1710, a1711,
                        a1712, a1713, a1714, a1715, c18, a1801,
                        a1808, a1809, a1810, a1811, a1812, a1813, a1814,
                        a1815, a1817, c19, a1901, a1908, a1909,
                        a1910, a1911, a1912, a1913, a1914, a1915, a1917,
                        a1918, c20, a2001, a2008, a2009, a2010,
                        a2011, a2012, a2013, a2014, a2015, a2017, a2018,
                        a2019, c21, a2101, a2108, a2109, a2110,
                        a2111, a2112, a2113, a2114, a2115, a2117, a2118,
                        a2119, a2120, c22, a2201, a2208, a2209,
                        a2210, a2211, a2212, a2213, a2214, a2215, a2217,
                        a2218, a2219, a2220, a2221, c23, a2301,
                        a2308, a2309, a2310, a2311, a2312, a2313,
                        a2314, a2315, a2317, a2318, a2319, a2320,
                        a2321, c24, a2401, a2408, a2409, a2410, a2411,
                        a2412, a2413, a2414, a2415, a2417,
                        a2418, a2419, a2420, a2421, c25, a2501, a2508,
                        a2509, a2510, a2511, a2512, a2513,
                        a2514, a2515, a2517, a2518, a2519, a2520,
                        a2521, c26, a2601, a2608, a2609, a2610,
                        a2611, a2612, a2613, a2614, a2615, a2617,
                        a2618, a2619, a2620, a2621 = tab.extra

                        k11 = f(
                            uprev +
                            dtold *
                            (a1701 * k1 + a1708 * k2 + a1709 * k3 + a1710 * k4 +
                             a1711 * k5 + a1712 * k6 + a1713 * k7 + a1714 * k8 +
                             a1715 * k9),
                            p, told + c17 * dtold)
                        k12 = f(
                            uprev +
                            dtold *
                            (a1801 * k1 + a1808 * k2 + a1809 * k3 + a1810 * k4 +
                             a1811 * k5 + a1812 * k6 + a1813 * k7 + a1814 * k8 +
                             a1815 * k9 + a1817 * k11),
                            p,
                            told + c18 * dtold)
                        k13 = f(
                            uprev +
                            dtold *
                            (a1901 * k1 + a1908 * k2 + a1909 * k3 + a1910 * k4 +
                             a1911 * k5 + a1912 * k6 + a1913 * k7 + a1914 * k8 +
                             a1915 * k9 + a1917 * k11 + a1918 * k12),
                            p,
                            told + c19 * dtold)
                        k14 = f(
                            uprev +
                            dtold *
                            (a2001 * k1 + a2008 * k2 + a2009 * k3 + a2010 * k4 +
                             a2011 * k5 + a2012 * k6 + a2013 * k7 + a2014 * k8 +
                             a2015 * k9 + a2017 * k11 + a2018 * k12 + a2019 * k13),
                            p,
                            told + c20 * dtold)
                        k15 = f(
                            uprev +
                            dtold *
                            (a2101 * k1 + a2108 * k2 + a2109 * k3 + a2110 * k4 +
                             a2111 * k5 + a2112 * k6 + a2113 * k7 + a2114 * k8 +
                             a2115 * k9 + a2117 * k11 + a2118 * k12 + a2119 * k13 +
                             a2120 * k14),
                            p,
                            told + c21 * dtold)
                        k16 = f(
                            uprev +
                            dtold *
                            (a2201 * k1 + a2208 * k2 + a2209 * k3 + a2210 * k4 +
                             a2211 * k5 + a2212 * k6 + a2213 * k7 + a2214 * k8 +
                             a2215 * k9 + a2217 * k11 + a2218 * k12 + a2219 * k13 +
                             a2220 * k14 + a2221 * k15),
                            p,
                            told + c22 * dtold)
                        k17 = f(
                            uprev +
                            dtold *
                            (a2301 * k1 + a2308 * k2 + a2309 * k3 + a2310 * k4 +
                             a2311 * k5 + a2312 * k6 + a2313 * k7 + a2314 * k8 +
                             a2315 * k9 + a2317 * k11 + a2318 * k12 + a2319 * k13 +
                             a2320 * k14 + a2321 * k15),
                            p,
                            told + c23 * dtold)
                        k18 = f(
                            uprev +
                            dtold *
                            (a2401 * k1 + a2408 * k2 + a2409 * k3 + a2410 * k4 +
                             a2411 * k5 + a2412 * k6 + a2413 * k7 + a2414 * k8 +
                             a2415 * k9 + a2417 * k11 + a2418 * k12 + a2419 * k13 +
                             a2420 * k14 + a2421 * k15),
                            p,
                            told + c24 * dtold)
                        k19 = f(
                            uprev +
                            dtold *
                            (a2501 * k1 + a2508 * k2 + a2509 * k3 + a2510 * k4 +
                             a2511 * k5 + a2512 * k6 + a2513 * k7 + a2514 * k8 +
                             a2515 * k9 + a2517 * k11 + a2518 * k12 + a2519 * k13 +
                             a2520 * k14 + a2521 * k15),
                            p,
                            told + c25 * dtold)
                        k20 = f(
                            uprev +
                            dtold *
                            (a2601 * k1 + a2608 * k2 + a2609 * k3 + a2610 * k4 +
                             a2611 * k5 + a2612 * k6 + a2613 * k7 + a2614 * k8 +
                             a2615 * k9 + a2617 * k11 + a2618 * k12 + a2619 * k13 +
                             a2620 * k14 + a2621 * k15),
                            p,
                            told + c26 * dtold)

                        us[cur_t] = uprev +
                                    dtold *
                                    (k1 * b1Θ + k2 * b8Θ + k3 * b9Θ + k4 * b10Θ +
                                     k5 * b11Θ +
                                     k6 * b12Θ + k7 * b13Θ + k8 * b14Θ + k9 * b15Θ +
                                     k11 * b17Θ +
                                     k12 * b18Θ + k13 * b19Θ + k14 * b20Θ + k15 * b21Θ +
                                     k16 * b22Θ +
                                     k17 * b23Θ + k18 * b24Θ + k19 * b25Θ + k20 * b26Θ)
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
