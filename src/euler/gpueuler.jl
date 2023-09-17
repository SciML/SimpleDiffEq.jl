#######################################################################################
# GPU-crutch solve method
# Makes the simplest possible method for GPU-compatibility
# Out of place only
#######################################################################################
struct GPUSimpleEuler <: AbstractSimpleDiffEqODEAlgorithm end
export GPUSimpleEuler

@muladd function DiffEqBase.solve(prob::ODEProblem,
    alg::GPUSimpleEuler;
    dt = error("dt is required for this algorithm"))
    @assert !isinplace(prob)
    u0 = prob.u0
    tspan = prob.tspan
    f = prob.f
    p = prob.p
    t = tspan[1]
    tf = prob.tspan[2]
    ts = tspan[1]:dt:tspan[2]
    us = MVector{length(ts), typeof(u0)}(undef)
    us[1] = u0
    u = u0

    for i in 2:length(ts)
        uprev = u
        t = ts[i]
        k1 = f(u, p, t)
        u = uprev + dt * k1
        us[i] = u
    end

    sol = DiffEqBase.build_solution(prob, alg, ts, SArray(us),
        k = nothing, stats = nothing,
        calculate_error = false)
    DiffEqBase.has_analytic(prob.f) &&
        DiffEqBase.calculate_solution_errors!(sol; timeseries_errors = true,
            dense_errors = false)
    sol
end
