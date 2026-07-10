using Documenter
using SimpleDiffEq

DocMeta.setdocmeta!(SimpleDiffEq, :DocTestSetup, :(using SimpleDiffEq); recursive = true)

makedocs(;
    modules = [SimpleDiffEq],
    sitename = "SimpleDiffEq.jl",
    format = Documenter.HTML(;
        prettyurls = get(ENV, "CI", "false") == "true",
        canonical = "https://docs.sciml.ai/SimpleDiffEq/stable/"
    ),
    pages = [
        "Home" => "index.md",
    ],
    checkdocs = :exports,
    warnonly = false
)
