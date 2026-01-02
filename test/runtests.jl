using SimpleDiffEq, SafeTestsets, Test

@time begin
    @time @safetestset "Explicit Imports Tests" include("explicit_imports_tests.jl")
    @time @safetestset "Discrete Tests" include("discrete_tests.jl")
    @time @safetestset "SimpleEM Tests" include("simpleem_tests.jl")
    @time @safetestset "SimpleTsit5 Tests" include("simpletsit5_tests.jl")
    @time @safetestset "SimpleATsit5 Tests" include("simpleatsit5_tests.jl")
    @time @safetestset "GPUSimpleATsit5 Tests" include("gpusimpleatsit5_tests.jl")
    @time @safetestset "SimpleRK4 Tests" include("simplerk4_tests.jl")
    @time @safetestset "SimpleEuler Tests" include("simpleeuler_tests.jl")
    @time @safetestset "GPU Compatible ODE Tests" include("gpu_ode_regression.jl")
    @time @safetestset "Interface Tests" include("interface_tests.jl")
end
