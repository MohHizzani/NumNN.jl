using Documenter, DocumenterTools

push!(LOAD_PATH,"../src/")
using NumNN

makedocs(
    sitename = "NumNN.jl",
    authors = "Mohammad Hizzani",
    # doctest   = false,
    # clean     = true,
    # linkcheck = false,
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true"
    ),
    modules = [NumNN],
    pages = Any[
        "Home" => "index.md",
        "Docstrings" => "docstrings.md",
        "Tutorials" => "tutorials.md"
    ],
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
#=deploydocs(
    repo = "<repository url>"
)=#

deploydocs(
    repo = "github.com/MohHizzani/NumNN.jl.git",
    target = "build",
    # push_preview = true,
    branch = "gh-pages",
    devurl = "dev",
    versions = ["stable" => "v^", "v#.#.#", devurl => devurl],
    # make = nothing,
    # forcepush = true,
)
