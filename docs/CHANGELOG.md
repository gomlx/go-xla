# Next: Dynamic Shapes (thx @ajroetker); added `Call()`; Quantized shapes;

* Updated `DefaultCPUVersion` to "v0.83.4" (`pjrt-cpu-binaries` version)
* Added `Call()` op (thx @ajroetker)
* Quantization:
  - Add Quantization field to shapes.Shape.
  - Add i2, i4, ui2 and ui4 DTypes.
  - Add UniformQuantize() and UniformDequantize() ops.
  - Add Value.WithOutputElementType() to allow change of quantization parameters for operations.
- Dynamic Shapes (thx @ajroetker)

# v0.1.4

* Updated installer library/cli to support linux/arm64 and windows/amd64. 
  Generalized CPU installation.
* linux and amazonlinux now use the same PJRT binary, built on a glibc-2.35 system (Ubuntu 22.04).
* Updated dependency to `pjrt-cpu-binaries` v0.83.3.

# v0.1.3

* Replaced GenPool by the `internal/pool.Pool`: it simplifies a bit, a bit faster and one less
  dependency.

# v0.1.2

* Removed external dependency to `github.com/charmbracelet/huh/spinner` from `pkg/installer`.
* Split `cmd/pjrt_installer` into its own module (its own `go.mod`), to limit default `go-xla` "apparent" dependencies.
* Package `pkg/installer`:
  * Spinner now limited to 1 line (truncates line to fit)

# v0.1.1

* Removed left-over debug message.

# v0.1.0 Merge `stablehlo` and `pjrt` into `go-xla`.

* Merge `stablehlo` and `pjrt` into `go-xla`.
  * Changed install directory to `.../lib/go-xla` (instead of `.../lib/gomlx/pjrt`).
  * Removed deprecated `xlabuilder`.
  * Split PJRT CPU binary releases into https://github.com/gomlx/pjrt-cpu-binaries
* Installation scripts also exported as a library.
* Improvements:
  * Fixed memory leaks on plugin destruction.
  * Replaced sync.Pool by GenPool: this avoids frequent unnecessary freeing of arenas.
* Added "installer.AutoInstall".
* Align with Google style:
  * Removed `github.com/janpfeifer/must` dependencies
  * Removed `testify` despendencies
