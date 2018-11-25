using BenchmarkTools, OrdinaryDiffEq
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
