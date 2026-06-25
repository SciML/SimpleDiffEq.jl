using SciMLTesting, SimpleDiffEq, Test
using JET

# `@..` is owned by FastBroadcast and the remaining ignored names are SciMLBase's
# solver-extension API; SimpleDiffEq imports/accesses both surfaces through
# DiffEqBase, which re-exports them. They are not public (exported / declared
# public) in their owner packages, so the owner/public ExplicitImports checks
# flag them. Ignore the specific names (they will become accepted as those base
# libraries mark them public).
const at = Symbol("@..")

# JET `report_package(; mode = :typo)` is clean on Julia 1.10/1.11 (JET 0.9.x) but
# surfaces 36 reports on Julia 1.12 (JET 0.11.x): one genuine latent bug
# (src/euler/euler.jl:199, an inplace-interpolation path that reads `length(u)`
# before `u` is assigned) and ~35 `may be undefined` reports on correlated-branch
# control flow (`cur_t` in the tsit5 `solve`/`gpuatsit5`/`gpuvern` loops, `q11` in
# the adaptive `step!` error-estimate blocks) that the earlier JET could prove safe
# but JET 0.11 cannot. The source is byte-identical to master, so these predate this
# QA conversion. Mark the JET check known-broken on 1.12+ (tracked in
# SciML/SimpleDiffEq.jl#123); the marker auto-flips to an Unexpected Pass when the
# reports are resolved. Gated by VERSION so the still-clean 1.10/1.11 lanes keep
# running the hard `test_package` check (a blanket `jet_broken` would record an
# Unexpected-Pass error there).
const jet_broken = VERSION >= v"1.12"

run_qa(
    SimpleDiffEq;
    explicit_imports = true,
    jet_broken = jet_broken,
    ei_kwargs = (;
        # owner is FastBroadcast (@..) / SciMLBase (the rest), imported via DiffEqBase
        all_explicit_imports_via_owners = (;
            ignore = (
                at, :AbstractODEAlgorithm, :AbstractODEIntegrator, :AbstractSDEAlgorithm,
                :ConstantInterpolation, :DEIntegrator, :__init, :__solve, :build_solution,
                :calculate_solution_errors!, :has_analytic, :is_diagonal_noise,
            ),
        ),
        all_qualified_accesses_via_owners = (;
            ignore = (
                at, :AbstractODEIntegrator, :AbstractSDEAlgorithm, :ConstantInterpolation,
                :DEIntegrator, :__init, :__solve, :build_solution, :calculate_solution_errors!,
                :has_analytic, :is_diagonal_noise,
            ),
        ),
        # non-public in DiffEqBase / SciMLBase / FastBroadcast
        all_qualified_accesses_are_public = (;
            ignore = (
                at, :AbstractODEIntegrator, :AbstractSDEAlgorithm, :ConstantInterpolation,
                :DEIntegrator, :ODE_DEFAULT_NORM, :__init, :__solve, :build_solution,
                :calculate_solution_errors!, :has_analytic, :is_diagonal_noise, :isadaptive,
            ),
        ),
        all_explicit_imports_are_public = (;
            ignore = (
                at, :AbstractODEAlgorithm, :AbstractODEIntegrator, :AbstractSDEAlgorithm,
                :ConstantInterpolation, :DEIntegrator, :ODE_DEFAULT_NORM, :__init, :__solve,
                :allows_arbitrary_number_types, :allowscomplex, :build_solution,
                :calculate_solution_errors!, :has_analytic, :is_diagonal_noise, :isadaptive,
                :isautodifferentiable,
            ),
        ),
    ),
)
