# XLA Backend for the `github.com/gomlx/compute` package.

Package xla implements a GoMLX backend, that is, a `github.com/gomlx/compute.Backend` interface using 
[Google's XLA (PJRT)](https://openxla.org/) as a backend.

The backend is registered with the aliases "xla", "stablehlo", "shlo" or "hlo" (all aliases to the same backend).

XLA/PJRT uses C++ written PJRT "plugins", `.so` files that implements XLA(PRJT). They are loaded dynamically in the program.

By default, the this XLA backend loads the requested plugins after the program starts and specifies the desired
plugin name (default to "cpu") using `dlopen`.

If the plugins are not available, the backend will download them automatically ("auto-install):

- From github.com/gomlx/pjrt-cpu-binaries for CPU PJRT plugins.
- From pypi.org, using the Jax pacakges for the CUDA and TPU PJRT plugins.

Auto-install has no effect if default plugins are already installed. But to control it you can:

  - Call xla.AutoInstall() directly if you want to call it immediately.
  - Configure it with xla.EnableAutoInstall() if you want to enable/disable it globally (default is enabled).
  - Set GOMLX_NO_AUTO_INSTALL, which sets the global auto-install flag to false -- but it can be overridden by
    calling xla.EnableAutoInstall().

Experimentally, one can get this backend to work with pre-linked PJRT plugins, but it will require the user to
add the `.so` files in a library in LD_LIBRARY_PATH, or precompile a `.a` static library.

  - Pre-link the CPU PJRT plugin statically: this will generate a bigger binary (+ ~200Mb, so slower to build),
    but allows one to build a static binary that can be deployed without extra dependencies (except the standard C and C++ libraries,
    usually available in most machines).
    To enable, build using the tag `pjrt_cpu_static` (e.g.: `go build --tags pjrt_cpu_static ...`),
    or import `github.com/gomlx/gomlx/backends/xla/cpu/static`. Both methods have the same effect.
  - Pre-link the CPU PJRT plugin dynamically: build with the build tag `pjrt_cpu_dynamic` (e.g.: `go test --tags pjrt_cpu_dynamic ...`),
    or import `github.com/gomlx/gomlx/backends/xla/cpu/dynamic`. Not much difference from linking the PJRT plugin
    after the program starts, as default.

## Shared Buffers Support:

XLA/PJRT for CPU allows the "device buffer" (where device=CPU) to be addressed directly, which
saves the copy from "host/local tensor" to the "on-device tensor" when executing a computation.
This is enabled by default if the plugin is called "cpu". To force advertising support for this
for other PJRTs provide the "shared_buffers" option, e.g.: GOMLX_BACKEND="xla:my_pjrt,shared_buffers".
Or to force disabling the support, provide the "noshared_buffers" option.

## Options

Those can be passed after the plugin name, e.g.: GOMLX_BACKEND="xla:my_pjrt,notf32,shared_buffers".

  - "tf32", "notf32": controls whether to use TF32 for DotGeneral operations that are using float32
    (it can be faster in modern GPUs). It's enabled by default.
  - "shared_buffers", "noshared_buffers": controls whether to use shared buffers for the device buffer
    (where device=CPU). It's enabled by default if the plugin is called "cpu".
