struct SimpleFunctionMap end
export SimpleFunctionMap

# ConstantCache version
function DiffEqBase.solve(prob::DiscreteProblem{uType,tType,false},
                          alg::SimpleFunctionMap;
                          calculate_values=true) where {uType,tType}
    tspan = prob.tspan
    f = prob.f
    u0 = prob.u0
    p = prob.p
    dt = 1

    n = Int((tspan[2] - tspan[1])/ dt + 1)
    u = [u0 for i in 1:n]
    t = tspan[1]:dt:tspan[2]
    if calculate_values
        for i in 2:n
            u[i] = f(u[i-1],p,t[i])
        end
    end
    sol = build_solution(prob,alg,t,u,dense=false,
                         interp = DiffEqBase.ConstantInterpolation(t,u),
                         calculate_error = false)
end

# Cache version
function DiffEqBase.solve(prob::DiscreteProblem{uType,tType,true},
                          alg::SimpleFunctionMap;
                          calculate_values=true) where {uType,tType}
    tspan = prob.tspan
    f = prob.f
    u0 = prob.u0
    p = prob.p
    dt = 1

    n = Int((tspan[2] - tspan[1])/ dt + 1)
    u = [similar(u0) for i in 1:n]
    u[1] .= u0
    t = tspan[1]:dt:tspan[2]
    if calculate_values
        for i in 2:n
            f(u[i],u[i-1],p,t[i])
        end
    end
    sol = build_solution(prob,alg,t,u,dense=false,
                         interp = DiffEqBase.ConstantInterpolation(t,u),
                         calculate_error = false)
end

##################################################

# Integrator version

mutable struct DiscreteIntegrator{F,uType,tType,P,S} <: DEIntegrator
    f::F
    u::uType
    t::tType
    uprev::uType
    p::P
    sol::S
    i::Int
end

function DiffEqBase.init(prob::DiscreteProblem,
                         alg::SimpleFunctionMap)
    sol = solve(prob,alg;calculate_values=false)
    DiscreteIntegrator(prob.f,prob.u0,prob.tspan[1],copy(prob.u0),prob.p,sol,1)
end

function step!(integrator::DiscreteIntegrator)
    integrator.t = integrator.i
    integrator.i += 1
    u = integrator.u
    uprev = integrator.uprev
    p = integrator.p
    f = integrator.f
    i = integrator.i

    if isinplace(integrator.sol.prob)
        f(integrator.sol.u[i],uprev,p,i)
        integrator.uprev = integrator.u
        integrator.u = integrator.sol.u[i]
    else
        u = f(uprev,p,i)
        integrator.sol.u[i] = u
        integrator.uprev = integrator.u
        integrator.u = u
    end
end
