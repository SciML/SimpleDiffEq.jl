using ExplicitImports
using SimpleDiffEq
using Test

@testset "ExplicitImports" begin
    @test check_no_implicit_imports(SimpleDiffEq) === nothing
    @test check_no_stale_explicit_imports(SimpleDiffEq) === nothing
end
