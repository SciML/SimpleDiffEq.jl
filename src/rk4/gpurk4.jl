#######################################################################################
# GPU-crutch solve method
# Makes the simplest possible method for GPU-compatibility
# Out of place only
#######################################################################################
struct GPUSimpleRK4 end
export GPUSimpleRK4

function DiffEqBase.solve(prob::ODEProblem,
                          alg::GPUSimpleRK4;
                          dt = error("dt is required for this algorithm"))
  @assert !isinplace(prob)
  u0 = prob.u0
  tspan = prob.tspan
  f = prob.f
  p = prob.p
  t = tspan[1]
  tf = prob.tspan[2]
  ts = tspan[1]:dt:tspan[2]
  us = MVector{length(ts),typeof(u0)}(undef)
  us[1] = u0
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
      us[i] = u
  end

  sol = DiffEqBase.build_solution(prob,alg,ts,SArray(us),
                                  k = nothing, destats = nothing,
                                  calculate_error = false)
  DiffEqBase.has_analytic(prob.f) && DiffEqBase.calculate_solution_errors!(sol;timeseries_errors=true,dense_errors=false)
  sol
end
