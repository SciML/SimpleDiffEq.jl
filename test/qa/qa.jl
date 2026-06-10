using SafeTestsets

@time @safetestset "ExplicitImports Tests" include("../explicit_imports_tests.jl")
@time @safetestset "JET Static Analysis Tests" include("../jet_tests.jl")
