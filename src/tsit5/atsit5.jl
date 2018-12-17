using RecursiveArrayTools: recursivecopy

struct SimpleATsit5 end

# PI-adaptive stepping parameters:
const beta1 = 7/50
const beta2 = 2/25
const qmax = 10.0
const qmin = 1/5
const gamma = 9/10
const qoldinit = 1e-4

defaultnorm(tmp) = @fastmath sqrt(sum(abs2,tmp)/length(tmp))
function defaultnorm(tmp::Vector{<:AbstractVector{T}}) where {T<:Number}
    x = zero(T)
    M = length(tmp)
    @inbounds for j in 1:M
        @fastmath x += sqrt(sum(abs2,tmp[j])/length(tmp[j]))
    end
    x/M
end

mutable struct SimpleATsit5Integrator{IIP, S, T, P, F, N} <: DiffEqBase.DEIntegrator
    f::F                  # eom
    uprev::S              # previous state
    u::S                  # current state
    tmp::S                # dummy, same as state
    tprev::T              # previous time
    t::T                  # current time
    t0::T                 # initial time, only for reinit
    tf::T
    dt::T                 # step size
    p::P                  # parameter container
    u_modified::Bool
    ks::Vector{S}         # interpolants of the algorithm
    cs::SVector{6, T}     # ci factors cache: time coefficients
    as::SVector{21, T}    # aij factors cache: solution coefficients
    btildes::SVector{7,T}
    rs::SVector{22, T}    # rij factors cache: interpolation coefficients
    qold::Float64
    abstol::Float64
    reltol::Float64
    internalnorm::N       # function that computes the error EEst based on state
end
const SAT5I = SimpleATsit5Integrator

DiffEqBase.isinplace(::SAT5I{IIP}) where {IIP} = IIP

#######################################################################################
# Initialization
#######################################################################################
function DiffEqBase.__init(prob::ODEProblem,alg::SimpleATsit5;
                         dt = error("dt is required for this algorithm"),
                         abstol = 1e-6, reltol = 1e-3,
                         internalnorm = defaultnorm)
  simpleatsit5_init(prob.f,DiffEqBase.isinplace(prob),prob.u0,
                   prob.tspan[1], prob.tspan[2], dt, prob.p, abstol, reltol,
                   internalnorm)
end

function DiffEqBase.__solve(prob::ODEProblem,alg::SimpleATsit5;
                          dt = error("dt is required for this algorithm"),
                          abstol = 1e-6, reltol = 1e-3,
                          internalnorm = defaultnorm)
  u0 = prob.u0
  tspan = prob.tspan
  ts = Vector{eltype(dt)}(undef,1)
  ts[1] = prob.tspan[1]
  us = Vector{typeof(u0)}(undef,0)
  push!(us,recursivecopy(u0))
  integ = simpleatsit5_init(prob.f,DiffEqBase.isinplace(prob),prob.u0,
                            tspan[1], tspan[2], dt, prob.p, abstol, reltol, internalnorm)
  # FSAL
  while integ.t < tspan[2]
    step!(integ)
    push!(us,recursivecopy(integ.u))
    push!(ts,integ.t)
  end
  sol = DiffEqBase.build_solution(prob,alg,ts,us,
                                  calculate_error = false)
  DiffEqBase.has_analytic(prob.f) && DiffEqBase.calculate_solution_errors!(sol;timeseries_errors=true,dense_errors=false)
  sol
end

function simpleatsit5_init(f::F,
                         IIP::Bool, u0::S, t0::T, tf::T, dt::T, p::P,
                         abstol, reltol,
                         internalnorm::N) where {F, P, S, T, N}

    cs, as, btildes, rs = _build_atsit5_caches(T)
    ks = _initialize_ks(u0)

    !IIP && @assert S <: SArray

    integ = SAT5I{IIP, S, T, P, F, N}(
        f, recursivecopy(u0), recursivecopy(u0), recursivecopy(u0), t0, t0, t0, tf, dt,
        p, true, ks, cs, as, btildes, rs,
        qoldinit,abstol,reltol, internalnorm
    )
end

_initialize_ks(u0::AbstractArray{T}) where {T<:Number} = [zero(u0) for i in 1:7]
function _initialize_ks(u0::Vector{<:AbstractVector})
    M = length(u0[1])
    for v in u0
        M != length(v) && error(
        "SimpleATsit5 supports Vector{Vector} stepping only when all inner "*
        "vectors are the same size."
        )
    end
    return [[zero(u0[j]) for j in 1:length(u0)] for i in 1:7]
end

function _build_atsit5_caches(::Type{T}) where {T}

    cs = SVector{6, T}(0.161, 0.327, 0.9, 0.9800255409045097, 1.0, 1.0)

    as = SVector{21, T}(
        #=a21=# convert(T,0.161),
        #=a31=# convert(T,-0.008480655492356989),
        #=a32=# convert(T,0.335480655492357),
        #=a41=# convert(T,2.8971530571054935),
        #=a42=# convert(T,-6.359448489975075),
        #=a43=# convert(T,4.3622954328695815),
        #=a51=# convert(T,5.325864828439257),
        #=a52=# convert(T,-11.748883564062828),
        #=a53=# convert(T,7.4955393428898365),
        #=a54=# convert(T,-0.09249506636175525),
        #=a61=# convert(T,5.86145544294642),
        #=a62=# convert(T,-12.92096931784711),
        #=a63=# convert(T,8.159367898576159),
        #=a64=# convert(T,-0.071584973281401),
        #=a65=# convert(T,-0.028269050394068383),
        #=a71=# convert(T,0.09646076681806523),
        #=a72=# convert(T,0.01),
        #=a73=# convert(T,0.4798896504144996),
        #=a74=# convert(T,1.379008574103742),
        #=a75=# convert(T,-3.290069515436081),
        #=a76=# convert(T,2.324710524099774)
    )

    btildes = SVector{7,T}(
                convert(T,-0.00178001105222577714),
                convert(T,-0.0008164344596567469),
                convert(T,0.007880878010261995),
                convert(T,-0.1447110071732629),
                convert(T,0.5823571654525552),
                convert(T,-0.45808210592918697),
                convert(T,0.015151515151515152)
    )

    rs = SVector{22, T}(
        #=r11=# convert(T,1.0),
        #=r12=# convert(T,-2.763706197274826),
        #=r13=# convert(T,2.9132554618219126),
        #=r14=# convert(T,-1.0530884977290216),
        #=r22=# convert(T,0.13169999999999998),
        #=r23=# convert(T,-0.2234),
        #=r24=# convert(T,0.1017),
        #=r32=# convert(T,3.9302962368947516),
        #=r33=# convert(T,-5.941033872131505),
        #=r34=# convert(T,2.490627285651253),
        #=r42=# convert(T,-12.411077166933676),
        #=r43=# convert(T,30.33818863028232),
        #=r44=# convert(T,-16.548102889244902),
        #=r52=# convert(T,37.50931341651104),
        #=r53=# convert(T,-88.1789048947664),
        #=r54=# convert(T,47.37952196281928),
        #=r62=# convert(T,-27.896526289197286),
        #=r63=# convert(T,65.09189467479366),
        #=r64=# convert(T,-34.87065786149661),
        #=r72=# convert(T,1.5),
        #=r73=# convert(T,-4),
        #=r74=# convert(T,2.5),
    )

    return cs, as, btildes, rs
end

#######################################################################################
# IIP version for vectors and matrices
#######################################################################################
function DiffEqBase.step!(integ::SAT5I{true, S, T}) where {S, T}

    L = length(integ.u)

    c1, c2, c3, c4, c5, c6 = integ.cs;
    dt = integ.dt; t = integ.t; p = integ.p; tf = integ.tf
    a21, a31, a32, a41, a42, a43, a51, a52, a53, a54,
    a61, a62, a63, a64, a65, a71, a72, a73, a74, a75, a76 = integ.as
    btilde1, btilde2, btilde3, btilde4, btilde5, btilde6, btilde7 = integ.btildes

    k1, k2, k3, k4, k5, k6, k7 = integ.ks
    tmp = integ.tmp; f! = integ.f

    integ.uprev .= integ.u; uprev = integ.uprev; u = integ.u

    qold = integ.qold
    abstol = integ.abstol
    reltol = integ.reltol

    if integ.u_modified
      f!(k1, uprev, p, t)
      integ.u_modified=false
    else
      k1 .= k7
    end


    EEst = Inf

    @inbounds while EEst>1
      dt < 1e-14 && error("dt<dtmin")

      for i in 1:L
          tmp[i] = uprev[i]+dt*a21*k1[i]
      end
      f!(k2, tmp, p, t+c1*dt)
      for i in 1:L
          tmp[i] = uprev[i]+dt*(a31*k1[i]+a32*k2[i])
      end
      f!(k3, tmp, p, t+c2*dt)
      for i in 1:L
          tmp[i] = uprev[i]+dt*(a41*k1[i]+a42*k2[i]+a43*k3[i])
      end
      f!(k4, tmp, p, t+c3*dt)
      for i in 1:L
          tmp[i] = uprev[i]+dt*(a51*k1[i]+a52*k2[i]+a53*k3[i]+a54*k4[i])
      end
      f!(k5, tmp, p, t+c4*dt)
      for i in 1:L
          tmp[i] = uprev[i]+dt*(a61*k1[i]+a62*k2[i]+a63*k3[i]+a64*k4[i]+a65*k5[i])
      end
      f!(k6, tmp, p, t+dt)
      for i in 1:L
          u[i] = uprev[i]+dt*(a71*k1[i]+a72*k2[i]+a73*k3[i]+a74*k4[i]+a75*k5[i]+a76*k6[i])
      end
      f!(k7, u, p, t+dt)

      for i in 1:L
        tmp[i] = dt*(btilde1*k1[i]+btilde2*k2[i]+btilde3*k3[i]+btilde4*k4[i]+
                     btilde5*k5[i]+btilde6*k6[i]+btilde7*k7[i])
        tmp[i] = tmp[i]/(abstol+max(abs(uprev[i]),abs(u[i]))*reltol)
      end

      EEst = integ.internalnorm(tmp)

      if iszero(EEst)
        q = inv(qmax)
      else
        @fastmath q11 = EEst^beta1
        @fastmath q = q11/(qold^beta2)
      end

      if EEst > 1
        dt = dt/min(inv(qmin),q11/gamma)
      else # EEst <= 1
        @fastmath q = max(inv(qmax),min(inv(qmin),q/gamma))
        qold = max(EEst,qoldinit)
        dtold = dt
        dt = dt/q #dtnew
        dt = min(abs(dt),abs(tf-t-dtold))

        integ.dt = dt
        integ.qold = qold
        integ.tprev = t

        if (tf - t - dtold) < 1e-14
          integ.t = tf
        else
          integ.t += dtold
        end
      end
    end

    return  nothing
end

#######################################################################################
# OOP version for vectors and matrices
#######################################################################################
function DiffEqBase.step!(integ::SAT5I{false, S, T}) where {S, T}

    c1, c2, c3, c4, c5, c6 = integ.cs;
    dt = integ.dt; t = integ.t; p = integ.p; tf = integ.tf
    a21, a31, a32, a41, a42, a43, a51, a52, a53, a54,
    a61, a62, a63, a64, a65, a71, a72, a73, a74, a75, a76 = integ.as
    btilde1, btilde2, btilde3, btilde4, btilde5, btilde6, btilde7 = integ.btildes

    tmp = integ.tmp; f = integ.f
    integ.uprev = integ.u; uprev = integ.u

    qold = integ.qold
    abstol = integ.abstol
    reltol = integ.reltol

    if integ.u_modified
      k1 = f(uprev, p, t)
      integ.u_modified=false
    else
      @inbounds k1 = integ.ks[7];
    end

    EEst = Inf

    while EEst > 1
      dt < 1e-14 && error("dt<dtmin")

      tmp = uprev+dt*a21*k1
      k2 = f(tmp, p, t+c1*dt)
      tmp = uprev+dt*(a31*k1+a32*k2)
      k3 = f(tmp, p, t+c2*dt)
      tmp = uprev+dt*(a41*k1+a42*k2+a43*k3)
      k4 = f(tmp, p, t+c3*dt)
      tmp = uprev+dt*(a51*k1+a52*k2+a53*k3+a54*k4)
      k5 = f(tmp, p, t+c4*dt)
      tmp = uprev+dt*(a61*k1+a62*k2+a63*k3+a64*k4+a65*k5)
      k6 = f(tmp, p, t+dt)
      u = uprev+dt*(a71*k1+a72*k2+a73*k3+a74*k4+a75*k5+a76*k6)
      k7 = f(u, p, t+dt)

      tmp = dt*(btilde1*k1+btilde2*k2+btilde3*k3+btilde4*k4+
                   btilde5*k5+btilde6*k6+btilde7*k7)
      tmp = tmp./(abstol+max.(abs.(uprev),abs.(u))*reltol)
      EEst = integ.internalnorm(tmp)

      if iszero(EEst)
        q = inv(qmax)
      else
        @fastmath q11 = EEst^beta1
        @fastmath q = q11/(qold^beta2)
      end

      if EEst > 1
        dt = dt/min(inv(qmin),q11/gamma)
      else # EEst <= 1
        @fastmath q = max(inv(qmax),min(inv(qmin),q/gamma))
        qold = max(EEst,qoldinit)
        dtold = dt
        dt = dt/q #dtnew
        dt = min(abs(dt),abs(tf-t-dtold))

        @inbounds begin # Necessary for interpolation
            integ.ks[1] = k1; integ.ks[2] = k2; integ.ks[3] = k3
            integ.ks[4] = k4; integ.ks[5] = k5; integ.ks[6] = k6
            integ.ks[7] = k7;
        end

        integ.dt = dt
        integ.qold = qold
        integ.tprev = t
        integ.u = u

        if (tf - t - dtold) < 1e-14
          integ.t = tf
        else
          integ.t += dtold
        end
      end
    end

    return  nothing
end

#######################################################################################
# Vector of Vector (always in-place) stepping
#######################################################################################
# Vector{Vector}
function DiffEqBase.step!(integ::SAT5I{true, S, T}) where {S<:Vector{<:Array}, T}

    M = length(integ.u) # number of states
    L = length(integ.u[1])

    c1, c2, c3, c4, c5, c6 = integ.cs;
    dt = integ.dt; t = integ.t; p = integ.p; tf = integ.tf
    a21, a31, a32, a41, a42, a43, a51, a52, a53, a54,
    a61, a62, a63, a64, a65, a71, a72, a73, a74, a75, a76 = integ.as
    btilde1, btilde2, btilde3, btilde4, btilde5, btilde6, btilde7 = integ.btildes

    k1, k2, k3, k4, k5, k6, k7 = integ.ks
    tmp = integ.tmp; f! = integ.f

    @inbounds for j in 1:M
        integ.uprev[j] .= integ.u[j]
    end
    uprev = integ.uprev; u = integ.u

    qold = integ.qold; abstol = integ.abstol; reltol = integ.reltol

    # FSAL
    if integ.u_modified
        f!(k1, uprev, p, t)
        integ.u_modified=false
    else
        @inbounds for j in 1:M; k1[j] .= k7[j]; end
    end

    EEst = Inf

    @inbounds while EEst>1
        dt < 1e-14 && error("dt<dtmin")

        for j in 1:M
            for i in 1:L
                tmp[j][i] = uprev[j][i]+dt*a21*k1[j][i]
            end
        end

        f!(k2, tmp, p, t+c1*dt)
        for j in 1:M
            for i in 1:L
                tmp[j][i] = uprev[j][i]+dt*(a31*k1[j][i]+a32*k2[j][i])
            end
        end

        f!(k3, tmp, p, t+c2*dt)
        for j in 1:M
            for i in 1:L
                tmp[j][i] = uprev[j][i]+dt*(a41*k1[j][i]+a42*k2[j][i]+a43*k3[j][i])
            end
        end

        f!(k4, tmp, p, t+c3*dt)
        for j in 1:M
            for i in 1:L
                tmp[j][i] = uprev[j][i]+dt*(a51*k1[j][i]+a52*k2[j][i]+a53*k3[j][i]+a54*k4[j][i])
            end
        end

        f!(k5, tmp, p, t+c4*dt)
        for j in 1:M
            for i in 1:L
                tmp[j][i] = uprev[j][i]+dt*(a61*k1[j][i]+a62*k2[j][i]+a63*k3[j][i]+a64*k4[j][i]+a65*k5[j][i])
            end
        end

        f!(k6, tmp, p, t+dt)
        for j in 1:M
            for i in 1:L
                u[j][i] = uprev[j][i]+dt*(a71*k1[j][i]+a72*k2[j][i]+a73*k3[j][i]+a74*k4[j][i]+a75*k5[j][i]+a76*k6[j][i])
            end
        end

        f!(k7, u, p, t+dt)
        for j in 1:M
            for i in 1:L
                tmp[j][i] = dt*(btilde1*k1[j][i]+btilde2*k2[j][i]+btilde3*k3[j][i]+btilde4*k4[j][i]+
                            btilde5*k5[j][i]+btilde6*k6[j][i]+btilde7*k7[j][i])
                tmp[j][i] = tmp[j][i]/(abstol+max(abs(uprev[j][i]),abs(u[j][i]))*reltol)
            end
        end

        EEst = integ.internalnorm(tmp)

        if iszero(EEst)
            q = inv(qmax)
        else
            @fastmath q11 = EEst^beta1
            @fastmath q = q11/(qold^beta2)
        end

        if EEst > 1
            dt = dt/min(inv(qmin),q11/gamma)
        else # EEst <= 1
            @fastmath q = max(inv(qmax),min(inv(qmin),q/gamma))
            qold = max(EEst,qoldinit)
            dtold = dt
            dt = dt/q #dtnew
            dt = min(abs(dt),abs(tf-t-dtold))

            integ.dt = dt
            integ.qold = qold
            integ.tprev = t

            if (tf - t - dtold) < 1e-14
                integ.t = tf
            else
                integ.t += dtold
            end
        end
    end
    return  nothing
end

# Vector{SVector}
function DiffEqBase.step!(integ::SAT5I{true, S, T}) where {S<:Vector{<:SVector}, T}

    M = length(integ.u)
    L = length(integ.u[1])

    c1, c2, c3, c4, c5, c6 = integ.cs;
    dt = integ.dt; t = integ.t; p = integ.p; tf = integ.tf
    a21, a31, a32, a41, a42, a43, a51, a52, a53, a54,
    a61, a62, a63, a64, a65, a71, a72, a73, a74, a75, a76 = integ.as
    btilde1, btilde2, btilde3, btilde4, btilde5, btilde6, btilde7 = integ.btildes

    k1, k2, k3, k4, k5, k6, k7 = integ.ks
    tmp = integ.tmp; f! = integ.f

    integ.uprev .= integ.u; uprev = integ.uprev; u = integ.u

    qold = integ.qold
    abstol = integ.abstol
    reltol = integ.reltol

    @inbounds if integ.u_modified
        f!(k1, uprev, p, t)
        integ.u_modified=false
    else
        for j in 1:M
            k1[j] = k7[j]
        end
    end


    EEst = Inf

    @inbounds while EEst>1
        dt < 1e-14 && error("dt<dtmin")

        for j in 1:M
            tmp[j] = uprev[j]+dt*a21*k1[j]
        end

        f!(k2, tmp, p, t+c1*dt)
        for j in 1:M
            tmp[j] = uprev[j]+dt*(a31*k1[j]+a32*k2[j])
        end

        f!(k3, tmp, p, t+c2*dt)
        for j in 1:M
            tmp[j] = uprev[j]+dt*(a41*k1[j]+a42*k2[j]+a43*k3[j])
        end

        f!(k4, tmp, p, t+c3*dt)
        for j in 1:M
            tmp[j] = uprev[j]+dt*(a51*k1[j]+a52*k2[j]+a53*k3[j]+a54*k4[j])
        end

        f!(k5, tmp, p, t+c4*dt)
        for j in 1:M
            tmp[j] = uprev[j]+dt*(a61*k1[j]+a62*k2[j]+a63*k3[j]+a64*k4[j]+a65*k5[j])
        end

        f!(k6, tmp, p, t+dt)
        for j in 1:M
            u[j] = uprev[j]+dt*(a71*k1[j]+a72*k2[j]+a73*k3[j]+a74*k4[j]+a75*k5[j]+a76*k6[j])
        end

        f!(k7, u, p, t+dt)

        for j in 1:M
            tmp[j] = dt*(btilde1*k1[j]+btilde2*k2[j]+btilde3*k3[j]+btilde4*k4[j]+
                             btilde5*k5[j]+btilde6*k6[j]+btilde7*k7[j])
            tmp[j] = tmp[j]./(abstol+max.(abs.(uprev[j]),abs.(u[j]))*reltol)
        end


        EEst = integ.internalnorm(tmp)

        if iszero(EEst)
            q = inv(qmax)
        else
            @fastmath q11 = EEst^beta1
            @fastmath q = q11/(qold^beta2)
        end

        if EEst > 1
            dt = dt/min(inv(qmin),q11/gamma)
        else # EEst <= 1
            @fastmath q = max(inv(qmax),min(inv(qmin),q/gamma))
            qold = max(EEst,qoldinit)
            dtold = dt
            dt = dt/q #dtnew
            dt = min(abs(dt),abs(tf-t-dtold))

            integ.dt = dt
            integ.qold = qold
            integ.tprev = t

            if (tf - t - dtold) < 1e-14
                integ.t = tf
            else
                integ.t += dtold
            end
        end
    end
    return  nothing
end


#######################################################################################
# Interpolation
#######################################################################################
# Interpolation function, both OOP and IIP
function (integ::SAT5I{IIP, S, T})(t::Real) where {IIP, S, T<:AbstractArray{<:Number}}
    tnext, tprev, dt = integ.t, integ.tprev, integ.dt
    @assert tprev ≤ t ≤ tnext
    θ = (t - tprev)/dt
    b1θ, b2θ, b3θ, b4θ, b5θ, b6θ, b7θ = bθs(integ.rs, θ)

    ks = integ.ks
    if !IIP
        u = @inbounds integ.uprev + dt*(b1θ*ks[1] + b2θ*ks[2] + b3θ*ks[3] + b4θ*ks[4] +
                      b5θ*ks[5] + b6θ*ks[6] + b7θ*ks[7])
        return u
    else
        u = similar(integ.u)
        @inbounds for i in 1:length(u)
            u[i] = integ.uprev[i] + dt*(b1θ*ks[1][i] + b2θ*ks[2][i] + b3θ*ks[3][i] +
                   b4θ*ks[4][i] + b5θ*ks[5][i] + b6θ*ks[6][i] + b7θ*ks[7][i])
        end
        return u
    end
end

export SimpleATsit5
