struct SimpleTsit5 <: AbstractSimpleDiffEqODEAlgorithm end
export SimpleTsit5

mutable struct SimpleTsit5Integrator{IIP, S, T, P, F} <:
               DiffEqBase.AbstractODEIntegrator{SimpleTsit5, IIP, S, T}
    f::F                  # eom
    uprev::S              # previous state
    u::S                  # current state
    tmp::S                # dummy, same as state
    tprev::T              # previous time
    t::T                  # current time
    t0::T                 # initial time, only for reinit
    dt::T                 # step size
    tdir::T
    p::P                  # parameter container
    u_modified::Bool
    ks::Vector{S}         # interpolants of the algorithm
    cs::NTuple{6, T}     # ci factors cache: time coefficients
    as::NTuple{21, T}    # aij factors cache: solution coefficients
    rs::NTuple{22, T}    # rij factors cache: interpolation coefficients
end
const ST5I = SimpleTsit5Integrator

DiffEqBase.isinplace(::ST5I{IIP}) where {IIP} = IIP

#######################################################################################
# Initialization
#######################################################################################
function DiffEqBase.__init(prob::ODEProblem, alg::SimpleTsit5;
                           dt = error("dt is required for this algorithm"))
    simpletsit5_init(prob.f, DiffEqBase.isinplace(prob), prob.u0,
                     prob.tspan[1], dt, prob.p)
end

function DiffEqBase.__solve(prob::ODEProblem, alg::SimpleTsit5;
                            dt = error("dt is required for this algorithm"))
    u0 = prob.u0
    tspan = prob.tspan
    ts = Array(tspan[1]:dt:tspan[2])
    n = length(ts)
    us = Vector{typeof(u0)}(undef, n)
    @inbounds us[1] = _copy(u0)
    integ = simpletsit5_init(prob.f, DiffEqBase.isinplace(prob), prob.u0,
                             prob.tspan[1], dt, prob.p)
    # FSAL
    for i in 1:(n - 1)
        step!(integ)
        us[i + 1] = _copy(integ.u)
    end
    sol = DiffEqBase.build_solution(prob, alg, ts, us,
                                    calculate_error = false)
    DiffEqBase.has_analytic(prob.f) &&
        DiffEqBase.calculate_solution_errors!(sol; timeseries_errors = true,
                                              dense_errors = false)
    sol
end

@inline function simpletsit5_init(f::F,
                                  IIP::Bool, u0::S, t0::T, dt::T,
                                  p::P) where {F, P, T, S <: AbstractArray{T}}
    cs, as, rs = _build_tsit5_caches(T)
    ks = [zero(u0) for i in 1:7]

    !IIP && @assert S <: SArray

    integ = ST5I{IIP, S, T, P, F}(f, _copy(u0), _copy(u0), _copy(u0), t0, t0, t0, dt,
                                  sign(dt), p, true, ks, cs, as, rs)
end

@inline function _build_tsit5_caches(::Type{T}) where {T}
    cs = (0.161, 0.327, 0.9, 0.9800255409045097, 1.0, 1.0)

    as = (convert(T, 0.161),        #=a21=#
                        convert(T, -0.008480655492356989),        #=a31=#
                        convert(T, 0.335480655492357),        #=a32=#
                        convert(T, 2.8971530571054935),        #=a41=#
                        convert(T, -6.359448489975075),        #=a42=#
                        convert(T, 4.3622954328695815),        #=a43=#
                        convert(T, 5.325864828439257),        #=a51=#
                        convert(T, -11.748883564062828),        #=a52=#
                        convert(T, 7.4955393428898365),        #=a53=#
                        convert(T, -0.09249506636175525),        #=a54=#
                        convert(T, 5.86145544294642),        #=a61=#
                        convert(T, -12.92096931784711),        #=a62=#
                        convert(T, 8.159367898576159),        #=a63=#
                        convert(T, -0.071584973281401),        #=a64=#
                        convert(T, -0.028269050394068383),        #=a65=#
                        convert(T, 0.09646076681806523),        #=a71=#
                        convert(T, 0.01),        #=a72=#
                        convert(T, 0.4798896504144996),        #=a73=#
                        convert(T, 1.379008574103742),        #=a74=#
                        convert(T, -3.290069515436081),        #=a75=#
                        convert(T, 2.324710524099774))

    rs = (convert(T, 1.0),        #=r11=#
                        convert(T, -2.763706197274826),        #=r12=#
                        convert(T, 2.9132554618219126),        #=r13=#
                        convert(T, -1.0530884977290216),        #=r14=#
                        convert(T, 0.13169999999999998),        #=r22=#
                        convert(T, -0.2234),        #=r23=#
                        convert(T, 0.1017),        #=r24=#
                        convert(T, 3.9302962368947516),        #=r32=#
                        convert(T, -5.941033872131505),        #=r33=#
                        convert(T, 2.490627285651253),        #=r34=#
                        convert(T, -12.411077166933676),        #=r42=#
                        convert(T, 30.33818863028232),        #=r43=#
                        convert(T, -16.548102889244902),        #=r44=#
                        convert(T, 37.50931341651104),        #=r52=#
                        convert(T, -88.1789048947664),        #=r53=#
                        convert(T, 47.37952196281928),        #=r54=#
                        convert(T, -27.896526289197286),        #=r62=#
                        convert(T, 65.09189467479366),        #=r63=#
                        convert(T, -34.87065786149661),        #=r64=#
                        convert(T, 1.5),        #=r72=#
                        convert(T, -4),        #=r73=#
                        convert(T, 2.5))

    return cs, as, rs
end

#######################################################################################
# Stepping
#######################################################################################
# IIP version for vectors and matrices
@inline @muladd function DiffEqBase.step!(integ::ST5I{true, S, T}) where {T, S}
    L = length(integ.u)

    c1, c2, c3, c4, c5, c6 = integ.cs
    dt = integ.dt
    t = integ.t
    p = integ.p
    a21, a31, a32, a41, a42, a43, a51, a52, a53, a54,
    a61, a62, a63, a64, a65, a71, a72, a73, a74, a75, a76 = integ.as

    k1, k2, k3, k4, k5, k6 = integ.ks
    k7 = k1
    tmp = integ.tmp
    f! = integ.f

    integ.uprev .= integ.u
    uprev = integ.uprev

    if integ.u_modified
        f!(k1, uprev, p, t)
        integ.u_modified = false
    else
        k1 .= k7
    end

    @inbounds begin
        for i in 1:L
            tmp[i] = uprev[i] + dt * a21 * k1[i]
        end
        f!(k2, tmp, p, t + c1 * dt)
        for i in 1:L
            tmp[i] = uprev[i] + dt * (a31 * k1[i] + a32 * k2[i])
        end
        f!(k3, tmp, p, t + c2 * dt)
        for i in 1:L
            tmp[i] = uprev[i] + dt * (a41 * k1[i] + a42 * k2[i] + a43 * k3[i])
        end
        f!(k4, tmp, p, t + c3 * dt)
        for i in 1:L
            tmp[i] = uprev[i] + dt * (a51 * k1[i] + a52 * k2[i] + a53 * k3[i] + a54 * k4[i])
        end
        f!(k5, tmp, p, t + c4 * dt)
        for i in 1:L
            tmp[i] = uprev[i] +
                     dt *
                     (a61 * k1[i] + a62 * k2[i] + a63 * k3[i] + a64 * k4[i] + a65 * k5[i])
        end
        f!(k6, tmp, p, t + dt)
        for i in 1:L
            integ.u[i] = uprev[i] +
                         dt * (a71 * k1[i] + a72 * k2[i] + a73 * k3[i] + a74 * k4[i] +
                          a75 * k5[i] + a76 * k6[i])
        end
    end
    f!(k7, integ.u, p, t + dt)

    integ.tprev = t
    integ.t += dt

    return nothing
end

# OOP version for vectors and matrices
@inline @muladd function DiffEqBase.step!(integ::ST5I{false, S, T}) where {T, S}
    c1, c2, c3, c4, c5, c6 = integ.cs
    dt = integ.dt
    t = integ.t
    p = integ.p
    a21, a31, a32, a41, a42, a43, a51, a52, a53, a54,
    a61, a62, a63, a64, a65, a71, a72, a73, a74, a75, a76 = integ.as

    tmp = integ.tmp
    f = integ.f
    integ.uprev = integ.u
    uprev = integ.u

    if integ.u_modified
        k1 = f(uprev, p, t)
        integ.u_modified = false
    else
        @inbounds k1 = integ.ks[7]
    end

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

    integ.u = uprev +
              dt * ((a71 * k1 + a72 * k2 + a73 * k3 + a74 * k4) + a75 * k5 + a76 * k6)
    k7 = f(integ.u, p, t + dt)

    @inbounds begin # Necessary for interpolation
        integ.ks[1] = k7
        integ.ks[2] = k2
        integ.ks[3] = k3
        integ.ks[4] = k4
        integ.ks[5] = k5
        integ.ks[6] = k6
        integ.ks[7] = k7
    end

    integ.tprev = t
    integ.t += dt

    return nothing
end

#######################################################################################
# Interpolation
#######################################################################################
# Interpolation function, OOP
@muladd function (integ::ST5I)(t::T) where {T}
    tnext, tprev, dt = integ.t, integ.tprev, integ.dt
    #@assert tprev ≤ t ≤ tnext
    θ = (t - tprev) / dt
    b1θ, b2θ, b3θ, b4θ, b5θ, b6θ, b7θ = bθs(integ.rs, θ)

    ks = integ.ks
    if !isinplace(integ)
        u = @inbounds integ.uprev +
                      dt * (b1θ * ks[1] + b2θ * ks[2] + b3θ * ks[3] + b4θ * ks[4] +
                       b5θ * ks[5] + b6θ * ks[6] + b7θ * ks[7])
        return u
    else
        u = similar(integ.u)
        @inbounds for i in 1:length(u)
            u[i] = integ.uprev[i] +
                   dt * (b1θ * ks[1][i] + b2θ * ks[2][i] + b3θ * ks[3][i] +
                    b4θ * ks[4][i] + b5θ * ks[5][i] + b6θ * ks[6][i] + b7θ * ks[7][i])
        end
        return u
    end
end
# Interpolation coefficients
@inline function bθs(rs, θ)
    # θ in (0, 1) !
    r11, r12, r13, r14, r22, r23, r24, r32, r33, r34, r42, r43, r44, r52, r53,
    r54, r62, r63, r64, r72, r73, r74 = rs

    b1θ::T = @evalpoly(θ, zero(T), r11, r12, r13, r14)
    b2θ::T = @evalpoly(θ, zero(T), zero(T), r22, r23, r24)
    b3θ::T = @evalpoly(θ, zero(T), zero(T), r32, r33, r34)
    b4θ::T = @evalpoly(θ, zero(T), zero(T), r42, r43, r44)
    b5θ::T = @evalpoly(θ, zero(T), zero(T), r52, r53, r54)
    b6θ::T = @evalpoly(θ, zero(T), zero(T), r62, r63, r64)
    b7θ::T = @evalpoly(θ, zero(T), zero(T), r72, r73, r74)

    return b1θ, b2θ, b3θ, b4θ, b5θ, b6θ, b7θ
end
