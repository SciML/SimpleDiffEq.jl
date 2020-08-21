#######################################################################################
# Simplest Loop method
# Makes the simplest possible method for teaching and performance testing
#######################################################################################
struct LoopRK4 <: DiffEqBase.AbstractODEAlgorithm end
export LoopRK4

# Out-of-place
# No caching, good for static arrays, bad for arrays
function DiffEqBase.__solve(prob::ODEProblem{uType,tType,false},
                          alg::LoopRK4;
                          dt = error("dt is required for this algorithm"),
                          save_everystep = true,
                          save_start = true,
                          adaptive = false,
                          dense = false,
                          save_end = true,
                          unstable_check = nothing) where {uType,tType}
  @assert !adaptive
  @assert !dense
  u0 = prob.u0
  tspan = prob.tspan
  f = prob.f
  p = prob.p
  t = tspan[1]
  tf = prob.tspan[2]
  ts = tspan[1]:dt:tspan[2]

  if save_everystep && save_start
      us = Vector{typeof(u0)}(undef,length(ts))
      us[1] = u0
  elseif save_everystep
      us = Vector{typeof(u0)}(undef,length(ts)-1)
  elseif save_start
      us = Vector{typeof(u0)}(undef,2)
      us[1] = u0
  else
      us = Vector{typeof(u0)}(undef,1) # for interface compatibility
  end

  u = u0
  half = convert(eltype(u0),1//2)
  sixth = convert(eltype(u0),1//6)

  for i in 2:length(ts)
      uprev = u; t = ts[i]
      k1 = f(u,p,t)
      tmp = uprev+dt*half*k1
      k2 = f(tmp, p, t+half*dt)
      tmp = uprev+dt*half*k2
      k3 = f(tmp, p, t+half*dt)
      tmp = uprev+dt*k3
      k4 = f(tmp, p, t+dt)
      u = uprev+dt*sixth*(k1+2k2+2k3+k4)
      save_everystep && (us[i] = u)
  end

  !save_everystep && save_end && (us[end] = u)

  sol = DiffEqBase.build_solution(prob,alg,ts,us,
                                  k = nothing, destats = nothing,
                                  calculate_error = false)
  DiffEqBase.has_analytic(prob.f) && DiffEqBase.calculate_solution_errors!(sol;timeseries_errors=true,dense_errors=false)
  sol
end

# In-place
# Good for mutable objects like arrays
# Use DiffEqBase.@.. for simd ivdep
function DiffEqBase.solve(prob::ODEProblem{uType,tType,true},
                          alg::LoopRK4;
                          dt = error("dt is required for this algorithm"),
                          save_everystep = true,
                          save_start = true,
                          adaptive = false,
                          dense = false,
                          save_end = true,
                          unstable_check = nothing) where {uType,tType}
  @assert !adaptive
  @assert !dense
  u0 = prob.u0
  tspan = prob.tspan
  f = prob.f
  p = prob.p
  t = tspan[1]
  tf = prob.tspan[2]
  ts = tspan[1]:dt:tspan[2]

  if save_everystep && save_start
      us = Vector{typeof(u0)}(undef,length(ts))
      us[1] = u0
  elseif save_everystep
      us = Vector{typeof(u0)}(undef,length(ts)-1)
  elseif save_start
      us = Vector{typeof(u0)}(undef,2)
      us[1] = u0
  else
      us = Vector{typeof(u0)}(undef,1) # for interface compatibility
  end

  u = copy(u0); uprev = copy(u0)
  k1 = zero(u0); k2 = zero(u0); k3 = zero(u0); k4 = zero(u0)
  half = convert(eltype(u0),1//2)
  sixth = convert(eltype(u0),1//6)

  for i in 2:length(ts)
      uprev .= u; t = ts[i]
      f(k1,u,p,t)
      DiffEqBase.@.. u = uprev+dt*half*k1
      f(k2, u, p, t+half*dt)
      DiffEqBase.@.. u = uprev+dt*half*k2
      f(k3, u, p, t+half*dt)
      DiffEqBase.@.. u = uprev+dt*k3
      f(k4, u, p, t+dt)
      DiffEqBase.@.. u = uprev+dt*sixth*(k1+2k2+2k3+k4)
      save_everystep && (us[i] = copy(u))
  end

  !save_everystep && save_end && (us[end] = u)

  sol = DiffEqBase.build_solution(prob,alg,ts,us,
                                  k = nothing, destats = nothing,
                                  calculate_error = false)
  DiffEqBase.has_analytic(prob.f) && DiffEqBase.calculate_solution_errors!(sol;timeseries_errors=true,dense_errors=false)
  sol
end
