# go-xla
OpenXLA Go API Bindings: StableHLO and PJRT

# [XLA](https://openxla.org/)'s [StableHLO](https://openxla.org/stablehlo) Builder API for Go

[![GoDev](https://img.shields.io/badge/go.dev-reference-007d9c?logo=go&logoColor=white)](https://pkg.go.dev/github.com/gomlx/stablehlo?tab=doc)
[![GitHub](https://img.shields.io/github/license/gomlx/stablehlo)](https://github.com/gomlx/stablehlo/blob/main/LICENSE)
[![Go Report Card](https://goreportcard.com/badge/github.com/gomlx/stablehlo)](https://goreportcard.com/report/github.com/gomlx/stablehlo)
[![TestStatus](https://github.com/gomlx/stablehlo/actions/workflows/tests.yaml/badge.svg)](https://github.com/gomlx/gomlx/actions/workflows/tests.yaml)
[![Slack](https://img.shields.io/badge/Slack-GoMLX-purple.svg?logo=slack)](https://app.slack.com/client/T029RQSE6/C08TX33BX6U)
[![Sponsor gomlx](https://img.shields.io/badge/Sponsor-gomlx-white?logo=github&style=flat-square)](https://github.com/sponsors/gomlx)

> [!Note]
> Discussion in the [Slack channel #gomlx](https://app.slack.com/client/T029RQSE6/C08TX33BX6U)
> (you can [join the slack server here](https://invite.slack.golangbridge.org/))

<img align="right" src="docs/gomlx_stablehlo_gopher.png" alt="GoMLX Gopher" width="220px"/>

[StableHLO](https://openxla.org/stablehlo) is an operation set for high-level operations (HLO) in machine learning (ML) models. 

It's the portability layer between ML frameworks (targeted for GoMLX, but could be used for others) and ML
compilers. It allows for easy support for different vendors, by coupling with **XLA's PJRT** (*) API for executing
StableHLO programs. So many different GPUs and TPUs are supported.

(*) **PJRT**, which stands for Pluggable JIT Runtime, is an API in the context of XLA (Accelerated Linear Algebra)
that provides a unified, cross-platform interface for interacting with different hardware accelerators. 
StableHLO is the device-independent language to specify the computation, and it also includes APIs to handle
buffer (the data) management and optionally distributed execution.

See:

* [StableHLO specification](https://openxla.org/stablehlo/spec)
* [GoMLX](https://github.com/gomlx/gomlx): a Go ML framework that supports an XLA (StableHLO+PJRT) backend to
  efficiently run (or train) ML programs.
* [GoPJRT](https://github.com/gomlx/gopjrt): a Go wrapper for PJRT C API, capable of executing StableHLO programs,
  for a lower level API.

## Examples

The tests in [`tests/gopjrt/gopjrt_test.go`](https://github.com/gomlx/stablehlo/blob/main/tests/gopjrt/gopjrt_test.go) 
should serve as simple examples of each operation.

Notice that `stablehlo` is a low-level API, usually used to build higher-level frameworks (an ML framework like GoMLX, 
maybe an image manipulation library that uses accelerators like GPUs, some scientific library, etc.), so it's deliberately 
verbose and requires boilerplate (error handling) everywhere. 
It sacrifices ergonomics for performance, consistency and stability. 

See another example of `stablehlo` and GoPJRT (to execute the generate StableHLO program) in [Mandelbrot mandelbrot.ipynb notebook](https://github.com/gomlx/gopjrt/blob/main/examples/mandelbrot.ipynb).
It includes some sample StableHLO code, if you are curious.

<a href="https://github.com/gomlx/gopjrt/blob/main/examples/mandelbrot.ipynb">
<img src="https://github.com/gomlx/gopjrt/assets/7460115/d7100980-e731-438d-961e-711f04d4425e" style="width:400px; height:240px"/>
</a>

## Status of Operations

Most operations are already implemented. See the [list of supported operations](https://github.com/gomlx/stablehlo/blob/main/internal/optypes/optypes.go#L91)
(the ones not implemented are in the bottom of the list).

If you need a specific operation, please open an issue.

See also the [CHANGELOG](https://github.com/gomlx/stablehlo/blob/main/docs/CHANGELOG.md).

## Dynamic Shapes Support: unbounded dynamism using shape polymorphism only!

* [Reference StableHLO documentation here](https://openxla.org/stablehlo/dynamism).
* [RFC: Dynamism 101](https://github.com/openxla/stablehlo/blob/main/rfcs/20230704-dynamism-101.md)

In the first version **we aim at supporting only _unbounded dynamism_ using shape polymorphism**:
where axes dimensions are not defined and has no bounds, and where PJRT will be able to dynamically
re-instantiate and re-compile the program to a new shape (or re-use a cache).

Other types of dynamism:

* _Unranked dynamism_: rank unknown and compile time. **Not supported**.
* _Data-dependent dynamism_: for data-dependent dynamic ops. For instance, if a function returns the indices of all 
  non-zero elements. **There is little support for this, so we are not support it yet.**

## StableHLO replaces [GoPJRT's XlaBuilder](https://github.com/gomlx/gopjrt/tree/main/xlabuilder)

With the following advantages:

* XlaBuilder has become a second-class citizen, so to say, within OpenXLA. 
  And things are moving towards the "MLIR builder" (MLIR is the generic ML Intermediary Language, of which StableHLO 
  is a specialization/extension).
  So we will eventually need a newer "builder" for **Gopjrt**.
* Since PJRT takes StableHLO in plain text format, we can write this entirely in Go, not requiring any extra
  C/C++ library build. 
  * PJRT itself is a C library, but with a relatively small API surface, and for which
    there are prebuilt distributions available (for Jax). So we can get away without having to manage Bazel issues.
  * The goal is to eventually not require a C compiler to compile gopjrt, and instead
    use [ebitengine/purego](https://github.com/ebitengine/purego) do dynamically load PJRT.
  * There are PJRT for different platforms. If we don't need to compile XlaBuilder for them, it makes it more plausible
    to support them.
 
The disadvantages:

* XlaBuilder provided "shape inference." So if I say `Add(a, b)` the XlaBuilder would tell how to broadcast
  a and b, and the resulting shape. When we build the StableHLO we have to re-implement this shape inference,
  not only for the `GoPJRT` users, but also because the *StableHLO* language requires the inputs and outputs shapes
  to be specified in every statement.
  * This is not a disadvantage for the user of this library, since `stablehlo` does that for you, but it's more
    work for the library maintainers.
* This means more maintenance: any updates in the language specification or new ops need to have their shape inference
  updated accordingly.

## The `shapeinference` sub-package

The same code is also used by [**GoMLX**](https://github.com/gomlx/gomlx) `SimpleGo` engine 
(`github.com/gomlx/gomlx/backends/simplego`), but we didn't want to create a dependency in either direction:
users of **Gopjrt** may not be interested in **GoMLX**, and users of **GoMLX** that don't use the XLA backend
wouldn't want a dependency to **Gopjrt**. 

Also, some operations have slightly different nuances.
