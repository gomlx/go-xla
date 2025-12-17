# go-xla: OpenXLA APIs bindings for Go

[![GoDev](https://img.shields.io/badge/go.dev-reference-007d9c?logo=go&logoColor=white)](https://pkg.go.dev/github.com/gomlx/go-xla?tab=doc)
[![GitHub](https://img.shields.io/github/license/gomlx/go-xla)](https://github.com/gomlx/go-xla/blob/master/LICENSE)
[![Go Report Card](https://goreportcard.com/badge/github.com/gomlx/go-xla)](https://goreportcard.com/report/github.com/gomlx/go-xla)
[![TestStatus](https://github.com/gomlx/go-xla/actions/workflows/linux_amd64_tests.yaml/badge.svg)](https://github.com/gomlx/go-xla/actions/workflows/linux_amd64_tests.yaml)
[![TestStatus](https://github.com/gomlx/go-xla/actions/workflows/linux_arm64_tests.yaml/badge.svg)](https://github.com/gomlx/go-xla/actions/workflows/linux_arm64_tests.yaml)
[![TestStatus](https://github.com/gomlx/go-xla/actions/workflows/darwin_tests.yaml/badge.svg)](https://github.com/gomlx/go-xla/actions/workflows/darwin_tests.yaml)
[![TestStatus](https://github.com/gomlx/go-xla/actions/workflows/windows_amd64_tests.yaml/badge.svg)](https://github.com/gomlx/go-xla/actions/workflows/windows_amd64_tests.yaml)
[![Slack](https://img.shields.io/badge/Slack-GoMLX-purple.svg?logo=slack)](https://app.slack.com/client/T029RQSE6/C08TX33BX6U)


## 🎯 Why use go-xla ?

<img align="right" width="220px" height="220px" alt="GoMLX Gopher" src="https://github.com/user-attachments/assets/346d8b1b-8bf0-4f95-938d-bc13fdca630b" />

The **go-xla** project leverages [OpenXLA's](https://openxla.org/) to (JIT-) compile, optimize, and **accelerate numeric computations**
(with large data) from Go using various [backends supported by OpenXLA](https://opensource.googleblog.com/2024/03/pjrt-plugin-to-accelerate-machine-learning.html): CPU, GPUs (Nvidia, AMD ROCm*, Intel*, Apple Metal*) and TPUs. 
It can be used to power Machine Learning frameworks (e.g. [GoMLX](https://github.com/gomlx/gomlx)), image processing, scientific 
computation, game AIs, etc. 

And because [Jax](https://docs.jax.dev/en/latest/), [TensorFlow](https://www.tensorflow.org/) and 
[optionally PyTorch](https://pytorch.org/xla/release/2.3/index.html) run on XLA, it is possible to run Jax functions in Go with **go-xla**, 
and probably TensorFlow and PyTorch as well.

The **go-xla** project aims to be minimalist and robust: it provides well-maintained, extensible Go wrappers for
[OpenXLA's StableHLO](https://openxla.org/#stablehlo) and [OpenXLA's PJRT](https://openxla.org/#pjrt). 

The APIs are not very "ergonomic" (error handling everywhere), but it's expected to be a stable building block for
other projects to create a friendlier API on top. 
The same way [Jax](https://jax.readthedocs.io/en/latest/) is a Python friendlier API on top of XLA/PJRT.

One such friendlier API co-developed with **go-xla** is [GoMLX, a Go machine learning framework](https://github.com/gomlx/gomlx).
But **go-xla** may be used as a standalone, for lower level access to XLA and other accelerator use cases—like running
Jax functions in Go, maybe an "accelerated" image processing or scientific simulation pipeline.

## 🧭 What is what?

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

The package [`github.com/gomlx/go-xla/pkg/stablehlo`](https://pkg.go.dev/github.com/gomlx/go-xla/pkg/stablehlo)
provides a Go API for writing StableHLO programs, including _shape inference_, needed to correctly 
infer the output shape of operations as the program is being built.

## 🗺️ How to use it?

Use the `stablehlo` to define your computation. Then use `pjrt` to compile (once) and execute it.

### [`github.com/gomlx/go-xla/pkg/stablehlo`](https://pkg.go.dev/github.com/gomlx/go-xla/pkg/pjrt)

* Create a `Builder` object with `stablehlo.NewBuilder()`
* Create a `Main` function with `Builder.Main()` (or other functions with `Builder.NewFunction()`)
* Define the function using the various operations defined in `stablehlo`. Their inputs and outputs are `*stablehlo.Value`
  types, which hold a reference to the `Function` they are defined in, as well as their `shapes.Shape`.
* Finish functions with `Function.Return(values...)`.
* Finish the _StableHLO_ program with `Builder.Build()`, it will return a string with the program, that can be fed to
  `pjrt` for compiling and execution.

Here is a sample of the `stablehlo` Go API, to create a module that calculates $f(x) = x^2+1$ (without the error handling lines):

```go
					builder := stablehlo.New("x_times_x_plus_1") // Use valid identifier for module name
					scalarF32 := shapes.Make(dtypes.F32)         // Scalar float32 shape
					mainFn := builder.Main()
					x, err := mainFn.NamedInput("x", scalarF32)
					fX, err := stablehlo.Multiply(x, x)
					one, err := mainFn.ConstantFromScalar(float32(1))
					fX, err = stablehlo.Add(fX, one)
					err = mainFn.Return(fX) // Set the return value for the main function
					stableHLOCode, err := builder.Build()
					fmt.Printf("StableHLO:\n%s\n", string(stableHLOCode))
```

Here is the _StableHLO_ text generated:

```mlir
module @x_times_x_plus_1 {
  func.func @main(%x: tensor<f32>) -> tensor<f32> {
    %0 = "stablehlo.multiply"(%x, %x) : (tensor<f32>, tensor<f32>) -> tensor<f32>
    %1 = "stablehlo.constant"() { value = dense<1.0> : tensor<f32> } : () -> tensor<f32>
    %2 = "stablehlo.add"(%0, %1) : (tensor<f32>, tensor<f32>) -> tensor<f32>
    "stablehlo.return"(%2) : (tensor<f32>) -> ()
  }
}
```

### [`github.com/gomlx/go-xla/pkg/pjrt`](https://pkg.go.dev/github.com/gomlx/go-xla/pkg/pjrt)

The `pjrt` package includes the following main concepts:

* `Plugin`: represents a PJRT plugin. It is created by calling `pjrt.GetPlugin(name)` (where `name` is the name of the plugin).
  It is the main entry point to the PJRT plugin.
* `Client`: first thing created after loading a plugin. It seems one can create a singleton `Client` per plugin,
  it's not very clear to me why one would create more than one `Client`.
* `LoadedExecutable`: Created when one calls `Client.Compile` a StableHLO program. The program is compiled and optimized
  to the PJRT target hardware and made ready to run.
* `Buffer`: Represents a buffer with the input/output data for the computations in the accelerators. There are 
  methods to transfer it to/from the host memory. They are the inputs and outputs of `LoadedExecutable.Execute`.

Example on how to run the computation with `pjrt`:

```go
var flagPluginName = flag.String("plugin", "cuda", "PRJT plugin name or full path")
...
err := installer.AutoInstall()  // Installs all supported plugin(s) if not already installed.
plugin, err := pjrt.GetPlugin(*flagPluginName)
client, err := plugin.NewClient(nil)
executor, err := client.Compile().WithStableHLO(stableHLOCode).Done()
for ii, value := range []float32{minX, minY, maxX, maxY} {
   inputs[ii], err = pjrt.ScalarToBuffer(m.client, value)
}
outputs, err := m.exec.Execute(inputs...).Done()
flat, err := pjrt.BufferToArray[float32](outputs[0])
outputs[0].Destroy() // Don't wait for the GC, destroy the buffer immediately.
...
```

For a more elaborate example, see [the mandelbrot.ipynb notebook](https://github.com/gomlx/go-xla/blob/main/examples/mandelbrot.ipynb),
which generates the image below:

<a href="https://github.com/gomlx/go-xla/blob/main/examples/mandelbrot.ipynb">
<img alt="Mandelbrot fractal figure" src="https://github.com/user-attachments/assets/6bf4d0bb-efe6-4b2e-8e8a-af2e34dbb63b" style="width:400px; height:240px"/>
</a>

## Installation of PJRT plugin

Most programs may simply add a call `installer.AutoInstall()` and it will automatically download the PJRT plugin
to the user's local home (`${HOME}/.local/lib/go-xla/` in Linux, `${HOME}/Library/Application Support/go-xla` in MacOS), 
if not installed already. 
It also auto-installs Nvidia PJRT plugin and required libraries if it's present.

To manually install it, or if you want a specific version, consider using the command line installer with 
`go run github.com/gomlx/go-xla/cmd/pjrt_installer@latest` and follow the
self-explanatory menu (or provide the flags for a quiet installation). 
Or run it with `go run github.com/gomlx/go-xla/cmd/pjrt_installer@latest -autoinstall`.


## 🤔 FAQ

* **When is feature X from PJRT going to be supported ?**
  The **go-xla** project doesn't wrap everything—although it does cover the most common operations. 
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

Environment variables that help control or debug how PJRT works:

* `PJRT_PLUGIN_LIBRARY_PATH`: Path to search for PJRT plugins. 
  The `pjrt` package also searches in `/usr/local/lib/go-xla`, `${HOME}/.local/lib/go-xla`, in
  the standard library paths for the system, and in the paths defined in `$LD_LIBRARY_PATH`.
  For compatibility with the older version, it also searches in 
  `/usr/local/lib/gomlx/pjrt`, `${HOME}/.local/lib/gomlx/pjrt`.
  In MacOS it's the equivalent under `${HOME}/Library/Application Support/`.
* `XLA_FLAGS`: Used by the C++ PJRT plugins. Documentation is linked by the [Jax XLA_FLAGS page](https://docs.jax.dev/en/latest/xla_flags.html),
  but I found it easier to just set this to "--help" and it prints out the flags.
* `XLA_DEBUG_OPTIONS`: If set, it is parsed as a `DebugOptions` proto that
  is passed during the JIT-compilation (`Client.Compile()`) of a computation graph.
  It is not documented how it works in PJRT (e.g., I observed a great slow down when this is set,
  even if set to the default values), but [the proto has some documentation](https://github.com/gomlx/go-xla/blob/main/protos/xla.proto#L40).

## For go-xla developers

* [Benchmarks on different strategies with CGO (Tab "Arena")](https://docs.google.com/spreadsheets/d/1ikpJH6rVVHq8ES-IA8U4lkKH4XsTSpRyZewXwGTgits/edit?gid=1008590518#gid=1008590518)
* [Execution benchmarks minimal programs (Tab "Basic XLA Metrics")](https://docs.google.com/spreadsheets/d/1ikpJH6rVVHq8ES-IA8U4lkKH4XsTSpRyZewXwGTgits/edit?gid=1369069161#gid=1369069161)

## 💖 Support the Project

If you find this project helpful, please consider 
[supporting our work through GitHub Sponsors](https://github.com/sponsors/gomlx). 

Your contribution helps us (currently mostly [me](https://github.com/janpfeifer)) 
maintain our coffee addiction :smiley: and dedicate more time to maintenance
and add new features for the entire GoMLX ecosystem.

It also helps us acquire access (buying or cloud) to hardware for more portability: e.g.: 
ROCm, Apple Metal (GPU), Multi-GPU/TPU, AWS Trainium, NVidia DGX Spark, Tenstorrent, etc.

## 💖 Acknowledgements

This project includes a (slightly modified) copy of the OpenXLA's [`pjrt_c_api.h`](https://github.com/openxla/xla/blob/main/xla/pjrt/c/pjrt_c_api.h) file as well as some of the `.proto` files used by `pjrt_c_api.h`.

More importantly, we **gratefully acknowledge the OpenXLA project and team** for their valuable work in developing and maintaining these plugins.

For more information about OpenXLA, please visit their website at [openxla.org](https://openxla.org/), or the GitHub page at [github.com/openxla/xla](https://github.com/openxla/xla)

## ⚖️ Licensing

> Copyright 2025 Jan Pfeifer

The **go-xla** project is [licensed under the Apache 2.0 license](https://github.com/gomlx/go-xla/blob/main/LICENSE).

The [OpenXLA project](https://openxla.org/), including `pjrt_c_api.h` file, the CPU and CUDA plugins, is [licensed under the Apache 2.0 license](https://github.com/openxla/xla/blob/main/LICENSE).

The CUDA plugin also uses the Nvidia CUDA Toolkit, which is subject to Nvidia's licensing terms and must be installed by the user or at the user's request.
