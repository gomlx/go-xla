# Notes For Developers

## Package `pjrt`

Lots is generated automatically `go generate ./...` from XLA's `pjrt_c_api.h` and its protos: you need to clone [github.com/openxla/xla](https://github.com/openxla/xla)
locally and set set the environment varialbe `XLA_SRC` to the path where XLA is cloned.

It includes a copy of the following files:

* `pjrt_c_api.h` from [github.com/openxla/xla/.../xla/pjrt/c/pjrt_c_api.h](https://github.com/openxla/xla/blob/main/xla/pjrt/c/pjrt_c_api.h), with the definitions of the PJRT plugin API.
  There is no easy way to integrate Go build system with Bazel (used by PJRT), so we just copied over the file (and mentioned it in the licensing).
* `compilation_options.proto`: ???

To generate the latest proto Go programs (see [tutorial](https://protobuf.dev/getting-started/gotutorial/)):
* Install the Protocol Buffers compiler: `sudo apt install protobuf-compiler`
* Install the most recent `protoc-gen-go`: `go install google.golang.org/protobuf/cmd/protoc-gen-go@latest`
* Set XLA_SRC to a clone of the `github.com/openxla/xla` repository.
* Go to the `protos` sub-package and do `go generate .` See `cmd/protoc_xla_prots/main.go` for details.

## PJRT Plugins

* A prebuilt CUDA (GPU) plugin is  [distributed with Jax (pypi wheel)](https://pypi.org/project/jax-cuda12-pjrt/) (albeit with a [non-standard naming](https://docs.google.com/document/d/1Qdptisz1tUPGn1qFAVgCV2omnfjN01zoQPwKLdlizas/edit#heading=h.l9ksu371j9wz))
* There is a prebuilt Apple Arm64+GPU metal plugin in [the jax-metal (pypi wheel)](https://pypi.org/project/jax-metal/) used by the installation script.
  It is lacking support for some functionality (including `float64`).
* The CPU plugin is built in repo [github.com/gomlx/pjrt-cpu-binaries](https://github.com/gomlx/pjrt-cpu-binaries), and the installer will download that
  the user's local library (`.local/lib/go-xla` in Linux, and the equivalent in Darwin/Windows).

## Updating `coverage.out` file

This is not done as a github actions because it would take too long to download the datasets, etc.
Instead, do it manually by running `cmd/run_coverage.sh` from the root of the repository.

