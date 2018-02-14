using StaticArrays, ForwardDiff, DiffEqBase
using DiffEqBase: isinplace
using OrdinaryDiffEq: FunctionMap
import DiffEqBase: init, step!, isinplace

export MinimalDiscreteProblem, MinimalDiscreteIntegrator
#####################################################################################
#                          Minimal Discrete Problem                                 #
#####################################################################################
"""
    MinimalDiscreteProblem(eom, state, p = nothing, t0 = 0)
"""
struct MinimalDiscreteProblem{IIP, F, S, P, D, T} <: DEProblem
    # D, T are dimension and eltype of state
    f::F      # eom, but same syntax as ODEProblem
    u0::S     # initial state
    p::P      # parameter container
    t0::Int   # initial time
end
MDP = MinimalDiscreteProblem
function MinimalDiscreteProblem(eom::F, state, p::P = nothing, t0 = 0) where {F, P}
    IIP = isinplace(eom, 4)
    # Ensure that there are only 2 cases: OOP with SVector or IIP with Vector
    # (requirement from ChaosTools)
    IIP || @assert typeof(eom(state, p, 0)) <: SVector
    u0 = IIP ? Vector(state) : SVector{length(state)}(state...)
    S = typeof(u0)
    D = length(u0); T = eltype(u0)
    MinimalDiscreteProblem{IIP, F, S, P, D, T}(eom, u0, p, t0)
end
isinplace(::MDP{IIP}) where {IIP} = IIP
mutable struct MinimalDiscreteIntegrator{IIP, F, S, P, D, T} <: DEIntegrator
    prob::MDP{IIP, F, S, P, D, T}
    u::S      # integrator state
    t::Int    # integrator "time" (counter)
    dummy::S  # dummy, used only in the IIP version
    p::P      # parameter container, I don't know why
end
MDI = MinimalDiscreteIntegrator
isinplace(::MDI{IIP}) where {IIP} = IIP

function init(prob::MDP{IIP, F, S, P, D, T},
    u = prob.u0) where {IIP, F, S, P, D, T}
    u0 = IIP ? Vector(u) : SVector{length(u)}(u...)
    return MDI{IIP, F, S, P, D, T}(prob, u0, prob.t0, deepcopy(u0), prob.p)
end

#####################################################################################
#                                   Stepping                                        #
#####################################################################################
# IIP version
function step!(integ::MDI{true})
    integ.dummy .= integ.u
    integ.prob.f(integ.u, integ.dummy, integ.p, integ.t)
    integ.t += 1
    return
end

function step!(integ::MDI{true}, N::Int)
    for i in 1:N
        integ.dummy .= integ.u
        integ.prob.f(integ.u, integ.dummy, integ.p, integ.t)
        integ.t += 1
    end
    return
end

# OOP version
step!(integ::MDI{false}) =
(integ.u = integ.prob.f(integ.u, integ.p, integ.t); integ.t +=1; nothing)
function step!(integ::MDI{false}, N::Int)
    for i in 1:N
        integ.u = integ.prob.f(integ.u, integ.p, integ.t)
        integ.t += 1
    end
    return
end



#####################################################################################
#                                    Tests                                          #
#####################################################################################
# Henon
@inline hoop(x, p, n) = SVector{2}(1.0 - p[1]*x[1]^2 + x[2], p[2]*x[1])
function hiip(dx, x, p, n)
    dx[1] = 1.0 - p[1]*x[1]^2 + x[2]
    dx[2] = p[2]*x[1]
    return
end
p = [1.4, 0.3]
u0 = zeros(2)

s1 = MDP(hoop, u0, p)
s2 = MDP(hiip, u0, p)

i1 = init(s1)
i2 = init(s2)

@assert isinplace(s1) == isinplace(i1) == false
@assert isinplace(s2) == isinplace(i2) == true

step!(i1); @assert i1.t == 1
step!(i2); @assert i2.t == 1

step!(i1, 1); step!(i2, 1)

@assert i1.u == i2.u
using BenchmarkTools

println("Time to apply functions (OOP/IIP)")
@btime $(s1.f)($s1.u0, $s1.p, $s1.t0)
a = rand(2)
@btime $(s2.f)($a, $s2.u0, $s2.p, $s2.t0)

println("Time to step (OOP/IIP), once")
@btime step!($i1)
@btime step!($i2)
println("Time to step (OOP/IIP), 1000s")
@btime step!($i1, 1000)
@btime step!($i2, 1000)
