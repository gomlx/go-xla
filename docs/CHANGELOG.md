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
