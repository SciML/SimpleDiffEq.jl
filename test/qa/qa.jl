using SciMLTesting, SimpleDiffEq, Test
using JET

run_qa(
    SimpleDiffEq;
    explicit_imports = true,
    api_docs_kwargs = (; rendered = true),
)
