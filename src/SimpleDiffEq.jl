__precompile__()

module SimpleDiffEq

using Reexport
@reexport using DiffEqBase

include("functionmap.jl")

end # module
