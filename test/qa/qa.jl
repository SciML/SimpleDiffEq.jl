using SciMLTesting, SimpleDiffEq, Test
using JET

function source_public_marked_names()
    marked_names = Symbol[]
    srcdir = normpath(joinpath(@__DIR__, "..", "..", "src"))

    for (root, _, files) in walkdir(srcdir), file in files
        endswith(file, ".jl") || continue

        for line in eachline(joinpath(root, file))
            stripped = strip(replace(line, r"#.*$" => ""))

            if startswith(stripped, "public ")
                for item in split(stripped[(lastindex("public ") + 1):end], ',')
                    m = match(r"^([A-Za-z_@][A-Za-z_0-9!]*)", strip(item))
                    m === nothing || push!(marked_names, Symbol(m.captures[1]))
                end
            end

            for m in eachmatch(r"(?:\bSciMLPublic\.)?@public\s+([A-Za-z_@][A-Za-z_0-9!]*)", stripped)
                push!(marked_names, Symbol(m.captures[1]))
            end
        end
    end

    return unique(marked_names)
end

function public_api_names(mod::Module)
    public_names = Symbol[]

    for name in names(mod; all = false, imported = true)
        name === nameof(mod) && continue

        if Base.isexported(mod, name) ||
                (isdefined(Base, :ispublic) && Base.ispublic(mod, name))
            push!(public_names, name)
        end
    end

    append!(public_names, source_public_marked_names())
    return sort!(unique(public_names); by = string)
end

function has_docstring(mod::Module, name::Symbol)
    doc = Base.Docs.doc(Base.Docs.Binding(mod, name))
    doc === nothing && return false

    text = sprint(show, MIME("text/plain"), doc)
    return !isempty(strip(text)) && !occursin("No documentation found", text)
end

function source_module_for(mod::Module, name::Symbol)
    value = getfield(mod, name)
    return value isa Module ? value : parentmodule(value)
end

function docs_entry_name(line)
    stripped = strip(line)
    isempty(stripped) && return nothing

    stripped = replace(stripped, r"\s+#.*$" => "")
    startswith(stripped, "SimpleDiffEq.") &&
        (stripped = stripped[(lastindex("SimpleDiffEq.") + 1):end])

    return Symbol(stripped)
end

function documented_api_names()
    documented_names = Symbol[]
    docs_src = normpath(joinpath(@__DIR__, "..", "..", "docs", "src"))

    for (root, _, files) in walkdir(docs_src), file in files
        endswith(file, ".md") || continue

        in_docs_block = false
        in_reexport_block = false
        for line in eachline(joinpath(root, file))
            stripped = strip(line)

            if startswith(stripped, "```@docs")
                in_docs_block = true
                continue
            elseif in_docs_block && startswith(stripped, "```")
                in_docs_block = false
                continue
            elseif startswith(stripped, "<!-- public-api-reexports-start -->")
                in_reexport_block = true
                continue
            elseif startswith(stripped, "<!-- public-api-reexports-end -->")
                in_reexport_block = false
                continue
            elseif in_docs_block
                name = docs_entry_name(stripped)
                name === nothing || push!(documented_names, name)
            elseif in_reexport_block
                m = match(r"^-\s+`([^`]+)`", stripped)
                m === nothing || push!(documented_names, Symbol(m.captures[1]))
            end
        end
    end

    return sort!(unique(documented_names); by = string)
end

@testset "public API documentation" begin
    public_names = public_api_names(SimpleDiffEq)
    documented_names = documented_api_names()

    missing_docstrings = filter(public_names) do name
        source_mod = source_module_for(SimpleDiffEq, name)
        !(has_docstring(source_mod, name) || has_docstring(SimpleDiffEq, name))
    end

    missing_docs_entries = setdiff(public_names, documented_names)

    @test isempty(missing_docstrings)
    @test isempty(missing_docs_entries)
end

run_qa(
    SimpleDiffEq;
    explicit_imports = true,
)
