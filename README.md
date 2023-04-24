# MNIST: Handwritten Digit Recognition in Julia

|                                 **Documentation**                                  |                                                                                                 **Build Status**                                                                                                 |                                        **Others**                                         |
| :--------------------------------------------------------------------------------: | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :---------------------------------------------------------------------------------------: |
| [![Stable][docs-stable-img]][docs-stable-url] [![Dev][docs-dev-img]][docs-dev-url] | [![Build Status][gha-img]][gha-url] [![Build Status][appveyor-img]][appveyor-url] [![Build Status][cirrus-img]][cirrus-url] [![pipeline status][gitlab-img]][gitlab-url] [![Coverage][codecov-img]][codecov-url] | [![GitHub license][license-img]][license-url] [![Code Style: Blue][style-img]][style-url] |

[docs-stable-img]: https://img.shields.io/badge/docs-stable-blue.svg
[docs-stable-url]: https://singularitti.github.io/MNIST.jl/stable
[docs-dev-img]: https://img.shields.io/badge/docs-dev-blue.svg
[docs-dev-url]: https://singularitti.github.io/MNIST.jl/dev
[gha-img]: https://github.com/singularitti/MNIST.jl/workflows/CI/badge.svg
[gha-url]: https://github.com/singularitti/MNIST.jl/actions
[appveyor-img]: https://ci.appveyor.com/api/projects/status/github/singularitti/MNIST.jl?svg=true
[appveyor-url]: https://ci.appveyor.com/project/singularitti/MNIST-jl
[cirrus-img]: https://api.cirrus-ci.com/github/singularitti/MNIST.jl.svg
[cirrus-url]: https://cirrus-ci.com/github/singularitti/MNIST.jl
[gitlab-img]: https://gitlab.com/singularitti/MNIST.jl/badges/main/pipeline.svg
[gitlab-url]: https://gitlab.com/singularitti/MNIST.jl/-/pipelines
[codecov-img]: https://codecov.io/gh/singularitti/MNIST.jl/branch/main/graph/badge.svg
[codecov-url]: https://codecov.io/gh/singularitti/MNIST.jl
[license-img]: https://img.shields.io/github/license/singularitti/MNIST.jl
[license-url]: https://github.com/singularitti/MNIST.jl/blob/main/LICENSE
[style-img]: https://img.shields.io/badge/code%20style-blue-4495d1.svg
[style-url]: https://github.com/invenia/BlueStyle

Welcome to MNIST.jl, a Julia package designed for recognizing handwritten digits using the
well-known MNIST dataset. This project aims to provide an efficient and user-friendly
interface for building, training, and evaluating machine learning models tailored for the
MNIST dataset.

The MNIST dataset is a collection of 70,000 labeled, 28x28 pixel images of handwritten
digits from 0 to 9. It has long served as a benchmark for machine learning algorithms,
particularly in the domain of image recognition. With MNIST.jl, you can easily access and
preprocess this dataset to train and evaluate your own models.

The images contained in the MNIST dataset are the property of Yann LeCun and Corinna Cortes.
We do not claim any copyright over the images. They have been acquired in their original IDX
format from http://yann.lecun.com/exdb/mnist/ and are stored in the data/ directory for your
convenience.

By leveraging the power of the Julia programming language, MNIST.jl offers a seamless and
efficient workflow for experimenting with various machine-learning techniques. We hope you
find this package useful in your journey to develop state-of-the-art handwritten digit
recognition models

The code is [hosted on GitHub](https://github.com/singularitti/MNIST.jl),
with some continuous integration services to test its validity.

This repository is created and maintained by [@singularitti](https://github.com/singularitti).
You are very welcome to contribute.

## Installation

The package can be installed with the Julia package manager.
From the Julia REPL, type `]` to enter the Pkg REPL mode and run:

```
pkg> add MNIST
```

Or, equivalently, via the [`Pkg` API](https://pkgdocs.julialang.org/v1/getting-started/):

```julia
julia> import Pkg; Pkg.add("MNIST")
```

## Documentation

- [**STABLE**][docs-stable-url] — **documentation of the most recently tagged version.**
- [**DEV**][docs-dev-url] — _documentation of the in-development version._

## Project status

The package is tested against, and being developed for, Julia `1.6` and above on Linux,
macOS, and Windows.

## Questions and contributions

You are welcome to post usage questions on [our discussion page][discussions-url].

Contributions are very welcome, as are feature requests and suggestions. Please open an
[issue][issues-url] if you encounter any problems. The [Contributing](@ref) page has
guidelines that should be followed when opening pull requests and contributing code.

[discussions-url]: https://github.com/singularitti/MNIST.jl/discussions
[issues-url]: https://github.com/singularitti/MNIST.jl/issues
