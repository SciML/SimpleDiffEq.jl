using SciMLTesting, SimpleDiffEq, Test
using JET

# `@..` is owned by FastBroadcast and the remaining ignored names are SciMLBase's
# solver-extension API; SimpleDiffEq imports/accesses both surfaces through
# DiffEqBase, which re-exports them. They are not public (exported / declared
# public) in their owner packages, so the owner/public ExplicitImports checks
# flag them. Ignore the specific names (they will become accepted as those base
# libraries mark them public).
const at = Symbol("@..")

run_qa(
    SimpleDiffEq;
    explicit_imports = true,
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
