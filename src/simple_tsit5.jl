struct SimpleTsit5 <: DiffEqBase.AbstractODEAlgorithm end
export SimpleTsit5

function DiffEqBase.solve(prob::ODEProblem,alg::SimpleTsit5,args...;
                          dt = error("dt is required for SimpleTsit5"), kwargs...)
  f! = prob.f
  u0 = prob.u0
  tspan = prob.tspan
  p = prob.p

  ts = Array(tspan[1]:dt:tspan[2])
  n = length(ts)
  us = [similar(u0) for i in 1:n]
  copyto!(us[1], u0)
  tmp = similar(u0)
  # FSAL
  ks = [similar(u0) for i in 1:6]
  f!(ks[1], u0, p, tspan[1])

  for i in 1:n-1
    uprev = us[i]
    t = ts[i]
    u = us[i+1]
    step!(u, uprev, f!, p, t, dt, tmp, ks)
  end

  sol = DiffEqBase.build_solution(prob,alg,ts,us,
                       calculate_error = false)
  DiffEqBase.has_analytic(f!) && DiffEqBase.calculate_solution_errors!(sol;timeseries_errors=true,dense_errors=false)
  sol
end

function step!(u::AbstractArray{T}, uprev, f!, p, t::T2, dt, tmp, ks) where {T, T2}
  c1  = convert(T2,0.161)
  c2  = convert(T2,0.327)
  c3  = convert(T2,0.9)
  c4  = convert(T2,0.9800255409045097)
  c5  = convert(T2,1)
  c6  = convert(T2,1)
  a21 = convert(T,0.161)
  a31 = convert(T,-0.008480655492356989)
  a32 = convert(T,0.335480655492357)
  a41 = convert(T,2.8971530571054935)
  a42 = convert(T,-6.359448489975075)
  a43 = convert(T,4.3622954328695815)
  a51 = convert(T,5.325864828439257)
  a52 = convert(T,-11.748883564062828)
  a53 = convert(T,7.4955393428898365)
  a54 = convert(T,-0.09249506636175525)
  a61 = convert(T,5.86145544294642)
  a62 = convert(T,-12.92096931784711)
  a63 = convert(T,8.159367898576159)
  a64 = convert(T,-0.071584973281401)
  a65 = convert(T,-0.028269050394068383)
  a71 = convert(T,0.09646076681806523)
  a72 = convert(T,0.01)
  a73 = convert(T,0.4798896504144996)
  a74 = convert(T,1.379008574103742)
  a75 = convert(T,-3.290069515436081)
  a76 = convert(T,2.324710524099774)
  #btilde1 = convert(T,-0.00178001105222577714)
  #btilde2 = convert(T,-0.0008164344596567469)
  #btilde3 = convert(T,0.007880878010261995)
  #btilde4 = convert(T,-0.1447110071732629)
  #btilde5 = convert(T,0.5823571654525552)
  #btilde6 = convert(T,-0.45808210592918697)
  #btilde7 = convert(T,0.015151515151515152)
  k1  = ks[1]
  k2  = ks[2]
  k3  = ks[3]
  k4  = ks[4]
  k5  = ks[5]
  k6  = ks[6]
  k7  = k1
  a = dt*a21
  @. tmp = uprev+a*k1
  f!(k2, tmp, p, t+c1*dt)
  @. tmp = uprev+dt*(a31*k1+a32*k2)
  f!(k3, tmp, p, t+c2*dt)
  @. tmp = uprev+dt*(a41*k1+a42*k2+a43*k3)
  f!(k4, tmp, p, t+c3*dt)
  @. tmp = uprev+dt*(a51*k1+a52*k2+a53*k3+a54*k4)
  f!(k5, tmp, p, t+c4*dt)
  @. tmp = uprev+dt*(a61*k1+a62*k2+a63*k3+a64*k4+a65*k5)
  f!(k6, tmp, p, t+dt)
  @. u = uprev+dt*(a71*k1+a72*k2+a73*k3+a74*k4+a75*k5+a76*k6)
  f!(k7, u, p, t+dt)
  nothing
end
