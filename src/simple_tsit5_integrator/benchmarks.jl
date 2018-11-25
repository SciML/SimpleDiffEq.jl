using BenchmarkTools, OrdinaryDiffEq

function loop(u, p, t)
    @inbounds begin
        σ = p[1]; ρ = p[2]; β = p[3]
        du1 = σ*(u[2]-u[1])
        du2 = u[1]*(ρ-u[3]) - u[2]
        du3 = u[1]*u[2] - β*u[3]
        return SVector{3}(du1, du2, du3)
    end
end
function liip(du, u, p, t)
    σ = p[1]; ρ = p[2]; β = p[3]
    du[1] = σ*(u[2]-u[1])
    du[2] = u[1]*(ρ-u[3]) - u[2]
    du[3] = u[1]*u[2] - β*u[3]
    return nothing
end


function bench()
    u0 = 10ones(3)
    dt = 0.01

    oop = init(MinimalTsit5(), loop, false, SVector{3}(u0), 0.0, dt, [10, 28, 8/3])
    step!(oop)

    iip = init(MinimalTsit5(), liip, true, u0, 0.0, dt, [10, 28, 8/3])
    step!(iip)

    odeoop = ODEProblem{false}(loop, SVector{3}(u0), (0.0, Inf),  [10, 28, 8/3])
    deoop = DiffEqBase.init(odeoop, Tsit5();
    adaptive = false, save_everystep = false, dt = dt, maxiters = Inf)
    step!(deoop)

    odeiip = ODEProblem{true}(liip, u0, (0.0, Inf),  [10, 28, 8/3])
    deiip = DiffEqBase.init(odeiip, Tsit5();
    adaptive = false, save_everystep = false, dt = dt, maxiters = Inf)

    step!(deiip)

    println("Minimal integrator times")
    println("In-place time")
    @btime step!($iip)

    println("Out of place time")
    @btime step!($oop)

    println("Standard Tsit5 times")
    println("In-place time")
    @btime step!($deiip)

    println("Out of place time")
    @btime step!($deoop)

end

bench()
