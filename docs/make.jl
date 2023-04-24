using MNIST
using Documenter

DocMeta.setdocmeta!(MNIST, :DocTestSetup, :(using MNIST); recursive=true)

makedocs(;
    modules=[MNIST],
    authors="singularitti <singularitti@outlook.com> and contributors",
    repo="https://github.com/singularitti/MNIST.jl/blob/{commit}{path}#{line}",
    sitename="MNIST.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://singularitti.github.io/MNIST.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/singularitti/MNIST.jl",
    devbranch="main",
)
