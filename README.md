# go-xla: OpenXLA APIs bindings for Go

[![GoDev](https://img.shields.io/badge/go.dev-reference-007d9c?logo=go&logoColor=white)](https://pkg.go.dev/github.com/gomlx/go-xla?tab=doc)
[![GitHub](https://img.shields.io/github/license/gomlx/go-xla)](https://github.com/gomlx/go-xla/blob/master/LICENSE)
[![Go Report Card](https://goreportcard.com/badge/github.com/gomlx/go-xla)](https://goreportcard.com/report/github.com/gomlx/go-xla)
[![TestStatus](https://github.com/gomlx/go-xla/actions/workflows/linux_tests.yaml/badge.svg)](https://github.com/gomlx/go-xla/actions/workflows/linux_tests.yaml)
[![TestStatus](https://github.com/gomlx/go-xla/actions/workflows/darwin_tests.yaml/badge.svg)](https://github.com/gomlx/go-xla/actions/workflows/darwin_tests.yaml)
[![Slack](https://img.shields.io/badge/Slack-GoMLX-purple.svg?logo=slack)](https://app.slack.com/client/T029RQSE6/C08TX33BX6U)



## Why use go-xla ?

The **go-xla** project leverages [OpenXLA's](https://openxla.org/) to (JIT-) compile, optimize, and **accelerate numeric computations**
(with large data) from Go using various [backends supported by OpenXLA](https://opensource.googleblog.com/2024/03/pjrt-plugin-to-accelerate-machine-learning.html): CPU, GPUs (Nvidia, AMD ROCm*, Intel*, Apple Metal*) and TPUs. 
It can be used to power Machine Learning frameworks (e.g. [GoMLX](https://github.com/gomlx/gomlx)), image processing, scientific 
computation, game AIs, etc. 

And because [Jax](https://docs.jax.dev/en/latest/), [TensorFlow](https://www.tensorflow.org/) and 
[optionally PyTorch](https://pytorch.org/xla/release/2.3/index.html) run on XLA, it is possible to run Jax functions in Go with GoPJRT, 
and probably TensorFlow and PyTorch as well.

The **go-xla** porject aims to be minimalist and robust: it provides well-maintained, extensible Go wrappers for
[OpenXLA's StableHLO](https://openxla.org/#stablehlo) and [OpenXLA's PJRT](https://openxla.org/#pjrt). 

The APIs are not very "ergonomic" (error handling everywhere), but it's expected to be a stable building block for
other projects to create a friendlier API on top. 
The same way [Jax](https://jax.readthedocs.io/en/latest/) is a Python friendlier API on top of XLA/PJRT.

One such friendlier API co-developed with **go-xla** is [GoMLX, a Go machine learning framework](https://github.com/gomlx/gomlx).
But **go-xla** may be used as a standalone, for lower level access to XLA and other accelerator use cases—like running
Jax functions in Go, maybe an "accelerated" image processing or scientific simulation pipeline.

## What is what?

### **PJRT** - "Pretty much Just another RunTime."

It is the heart of the OpenXLA project: it takes an IR (intermediate representation, typically _StableHLO_) of the "computation graph,"
JIT (Just-In-Time) compiles it (once) and executes it fast (many times). 
See the [Google's "PJRT: Simplifying ML Hardware and Framework Integration"](https://opensource.googleblog.com/2023/05/pjrt-simplifying-ml-hardware-and-framework-integration.html) blog post.

A "computation graph" is the part of your program (usually vectorial math/machine learning related) that one
wants to "accelerate." 

The PJRT comes in the form of a _plugin_, a dynamically linked library (`.so` file in Linux, or optionally 
`.dylib` in Darwin, or `.dll` in Windows). Typically, there is one plugin per hardware you are supporting. 
E.g.: there are PJRT plugins for CPU (Linux/amd64 and macOS for now, but likely it could be compiled for
other CPUs -- SIMD/AVX are well-supported), for TPUs (Google's accelerator), 
GPUs (Nvidia is well-supported; there are AMD and Intel's PJRT plugins, but they were not tested), 
and others are in development. Some PJRT plugins are not open-source, but are available for download.

The **go-xla** project provides the package `github.com/gomlx/go-xla/pkg/pjrt`, 
a Go API for dynamically loading and calling the **PJRT** runtime.
It also provides a installer or library (`github.com/gomlx/go-xla/pkg/installer`) to 
auto-install (download pre-compiled binaries) **PJRT** plugins for CPU (from GitHub), 
CUDA (from pypi.org Jax pacakges) and TPU (also from pypi.org).

### **StableHLO** - "Stable High Level Optimization" (?)

The currently better supported IR (intermediary representation) supported by PJRT, see specs 
in [StableHLO docs](https://openxla.org/stablehlo). It's a text representation of the computation
that can easily be parsed by computers, but not easily written or read by humans.

The package [`github.com/gomlx/go-xla/pkg/stablehlo`](https://github.com/gomlx/go-xla/pkg/stablehlo?tab=readme-ov-file)
provides a Go API for writing StableHLO programs, including _shape inference_, needed to correctly 
infer the output shape of operations as the program is being built.

is the current preferred IR language for XLA PJRT. This library (co-developed with **GoPJRT**) is a Go API for building
computation graphs in StableHLO that can be directly fed to *GoPJRT*. See examples below.
This is a wrapper Go library to an XLA C++ library that generates the previous IR (called MHLO).
It is still supported by XLA and by **GoPJRT**, but it is being deprecated.
3. Using Jax, Tensorflow, PyTorchXLA: Jax/Tensorflow/PyTorchXLA can output the StableHLO of JIT compiled functions  
that can be fed directly to PJRT (as text). We don't detail this here, but the authors did this a lot during
development of **GoPJRT**, [github.com/gomlx/go-xla/pkg/stablehlo](https://github.com/gomlx/go-xla/pkg/stablehlo?tab=readme-ov-file) and 
[github.com/gomlx/gopjtr/xlabuilder](https://github.com/gomlx/go-xla/tree/main/xlabuilder) for testing.

## Example

1. Minimalistic example, that assumes you have your StableHLO code in a variable (`[]byte`) called `stablehloCode`:

```go
var flagPluginName = flag.String("plugin", "cuda", "PRJT plugin name or full path")
...
plugin, err := pjrt.GetPlugin(*flagPluginName)
client, err := plugin.NewClient(nil)
executor, err := client.Compile().WithStableHLO(stablehloCode).Done()
for ii, value := range []float32{minX, minY, maxX, maxY} {
   inputs[ii], err = pjrt.ScalarToBuffer(m.client, value)
}
outputs, err := m.exec.Execute(inputs...).Done()
flat, err := pjrt.BufferToArray[float32](outputs[0])
outputs[0].Destroy() // Don't wait for the GC, destroy the buffer immediately.
...
```

2. See [mandelbrot.ipynb notebook](https://github.com/gomlx/go-xla/blob/main/examples/mandelbrot.ipynb) 
with an example building the computation for a Mandelbrot image using `stablehlo`, 
it includes a sample of the computation's StableHLO IR.

<a href="https://github.com/gomlx/go-xla/blob/main/examples/mandelbrot.ipynb">
<img alt="Mandelbrot fractal figure" src="https://github.com/gomlx/go-xla/assets/7460115/d7100980-e731-438d-961e-711f04d4425e" style="width:400px; height:240px"/>
</a>

## How to use it?

The main package is [`github.com/gomlx/go-xla/pkg/pjrt`](https://pkg.go.dev/github.com/gomlx/go-xla/pkg/pjrt), and we'll refer to it as simply `pjrt`.

The `pjrt` package includes the following main concepts:

* `Plugin`: represents a PJRT plugin. It is created by calling `pjrt.GetPlugin(name)` (where `name` is the name of the plugin).
  It is the main entry point to the PJRT plugin.
* `Client`: first thing created after loading a plugin. It seems one can create a singleton `Client` per plugin,
  it's not very clear to me why one would create more than one `Client`.
* `LoadedExecutable`: Created when one calls `Client.Compile` a StableHLO program. The program is compiled and optimized
  to the PJRT target hardware and made ready to run.
* `Buffer`: Represents a buffer with the input/output data for the computations in the accelerators. There are 
  methods to transfer it to/from the host memory. They are the inputs and outputs of `LoadedExecutable.Execute`.

## Installation of PJRT plugin

Most programs may simply add a call `installer.AutoInstall()` and it will automatically download the PJRT plugin
to the user's local home (`${HOME}/.local/lib/go-xla/` in Linux), if not installed already.
So there is nothing to do.

To manually install it, consider using the command line installer with 
`go run github.com/gomlx/go-xla/cmd/pjrt_installer@latest` and follow the
self-explanatory menu (or provide the flags for a quiet installation)

## FAQ

* **When is feature X from PJRT going to be supported ?**
  GoPJRT doesn't wrap everything—although it does cover the most common operations. 
  The simple ops and structs are auto-generated. But many require hand-writing.
  Please, if it is useful to your project, create an issue; I'm happy to add it. 
  I focus on the needs of GoMLX, but the idea is that it can serve other purposes, and I'm happy to support it.

* **Why does PJRT spit out so many logs? Can we disable it?**
  This is a great question ... imagine if every library we use decided they also want to clutter our stderr?
  I have [an open question in Abseil about it](https://github.com/abseil/abseil-cpp/discussions/1700).
  It may be some issue with [Abseil Logging](https://abseil.io/docs/python/guides/logging) which also has this other issue
  of not allowing two different linked programs/libraries to call its initialization (see [Issue #1656](https://github.com/abseil/abseil-cpp/issues/1656)).
  A hacky workaround is duplicating fd 2 and assign to Go's `os.Stderr`, and then close fd 2, so PJRT plugins
  won't have where to log. This hack is encoded in the function `pjrt.SuppressAbseilLoggingHack()`: call it
  before calling `pjrt.GetPlugin`. But it may have unintended consequences if some other library depends
  on the fd 2 to work, or if a real exceptional situation needs to be reported and is not.

## 🤝 Collaborating or asking for help

Discussion in the [Slack channel #gomlx](https://app.slack.com/client/T029RQSE6/C08TX33BX6U)
(you can [join the slack server here](https://invite.slack.golangbridge.org/)).


## Environment Variables

Environment variables that help control or debug how GoPJRT works:

* `PJRT_PLUGIN_LIBRARY_PATH`: Path to search for PJRT plugins. 
  GoPJRT also searches in `/usr/local/lib/gomlx/pjrt`, `${HOME}/.local/lib/gomlx/pjrt`, in
  the standard library paths for the system, and in the paths defined in `$LD_LIBRARY_PATH`.
* `XLA_FLAGS`: Used by the C++ PJRT plugins. Documentation is linked by the [Jax XLA_FLAGS page](https://docs.jax.dev/en/latest/xla_flags.html),
  but I found it easier to just set this to "--help" and it prints out the flags.
* `XLA_DEBUG_OPTIONS`: If set, it is parsed as a `DebugOptions` proto that
  is passed during the JIT-compilation (`Client.Compile()`) of a computation graph.
  It is not documented how it works in PJRT (e.g., I observed a great slow down when this is set,
  even if set to the default values), but [the proto has some documentation](https://github.com/gomlx/go-xla/blob/main/protos/xla.proto#L40).

## Acknowledgements

This project includes a (slightly modified) copy of the OpenXLA's [`pjrt_c_api.h`](https://github.com/openxla/xla/blob/main/xla/pjrt/c/pjrt_c_api.h) file as well as some of the `.proto` files used by `pjrt_c_api.h`.

More importantly, we **gratefully acknowledge the OpenXLA project and team** for their valuable work in developing and maintaining these plugins.

For more information about OpenXLA, please visit their website at [openxla.org](https://openxla.org/), or the GitHub page at [github.com/openxla/xla](https://github.com/openxla/xla)

## Licensing

The **go-xla** project is [licensed under the Apache 2.0 license](https://github.com/gomlx/go-xla/blob/main/LICENSE).

The [OpenXLA project](https://openxla.org/), including `pjrt_c_api.h` file, the CPU and CUDA plugins, is [licensed under the Apache 2.0 license](https://github.com/openxla/xla/blob/main/LICENSE).

The CUDA plugin also uses the Nvidia CUDA Toolkit, which is subject to Nvidia's licensing terms and must be installed by the user or at the user's request.
