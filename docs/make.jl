using Documenter
using NumNN

makedocs(
    sitename = "NumNN",
    format = Documenter.HTML(),
    modules = [NumNN]
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
#=deploydocs(
    repo = "<repository url>"
)=#
