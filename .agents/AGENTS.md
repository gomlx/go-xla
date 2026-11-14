# AGENTS.md — project documentation for AI coding agents

## Project Overview

`github.com/gomlx/go-xla` provides extensible Go wrappers for [OpenXLA's StableHLO](https://openxla.org/#stablehlo) and [OpenXLA's PJRT](https://openxla.org/#pjrt), an automatic PJRT plugin installer, and the `"xla"` backend implementation for [`github.com/gomlx/compute`](https://github.com/gomlx/compute).

### Relationship to `gomlx/compute` and `gomlx/gomlx`

- **[`github.com/gomlx/gomlx`](https://github.com/gomlx/gomlx)**: High-level ML framework in Go (tensors, layers, automatic differentiation, optimizers, model training loops). GoMLX delegates lower-level graph compilation and execution to a `compute.Backend`.
- **[`github.com/gomlx/compute`](https://github.com/gomlx/compute)**: Foundational execution and backend abstraction interface (`compute.Backend`, `compute.Builder`, `compute.Function`, `compute.Executable`, `compute.Buffer`), shared `dtypes`, `shapes`, `shapeinference`, and backend compliance tests (`compute/support/backendtest`).
- **`github.com/gomlx/go-xla` (This Repository)**: Depends **only** on `github.com/gomlx/compute` (never on `github.com/gomlx/gomlx`, to avoid circular dependencies) and is organized into four main layers:
  1. **`stablehlo` + `types`**: Pure-Go AST builder and serializer for StableHLO MLIR text programs, including eager shape inference, closures, Shardy distributed meshes/sharding, bounded dynamic shapes, and quantization metadata.
  2. **`pjrt`**: Low-level CGo wrapper around OpenXLA's C API (`pjrt_c_api.h`) for dynamically loading (`dlopen`) hardware PJRT plugins (`cpu`, `cuda`, `rocm`, `tpu`, etc.), compiling StableHLO/HLO modules, managing device memory, and executing computations.
  3. **`compute/xla`**: Implements `compute.Backend` (registered as `"xla"` with aliases `"stablehlo"`, `"shlo"`, `"hlo"`) by lowering `compute` operations into `stablehlo` and compiling/executing them through `pjrt`.
  4. **`installer` + `cmd/pjrt_installer`**: Library and standalone CLI for downloading and installing prebuilt PJRT plugins (CPU from `gomlx/pjrt-cpu-binaries`, CUDA and TPU from PyPI JAX wheels, ROCm from AMD's manylinux wheels).

When navigating the codebase of any of the related projects, read their .agents/AGENTS.md file to get more information about
the structure of the codebase and their specific conventions.

### Documentation & Git Workflow

- Record user-visible or API changes in `docs/CHANGELOG.md` (with `docs/CHANGELOG-pjrt.md` and `docs/CHANGELOG-stablehlo.md` kept as historical archives).
- Keep `.agents/AGENTS.md` (symlinked from `GEMINI.md`) updated whenever architecture, conventions, or non-obvious behavior changes.
- **Never `git commit` or `git push`** without explicit user instruction; leave changes uncommitted in the working tree for review.
- When changing code, if not explicitly asked, always wait for the operator to review the changes before committing them.

### Coding Style & Dependencies

- **Minimal External Dependencies (No Testify)**: Do **not** add external test libraries such as `github.com/stretchr/testify`. Use standard `testing` (`t.Fatalf`, `t.Errorf`, `t.Run`) and `github.com/gomlx/compute/support/testutil`. Note that `cmd/pjrt_installer` has its own `go.mod` specifically to isolate interactive TUI (`charmbracelet/*`) dependencies from the main `go-xla` module.
- **Error Handling**: Wrap errors with stack traces using `github.com/pkg/errors` (`errors.Errorf`, `errors.Wrapf`, `errors.WithMessagef`) and format with `"%+v"` when logging stack traces. In `compute/xla`, wrap unsupported ops or option combinations with `compute.ErrNotImplemented` so GoMLX and `backendtest` can fall back or skip cleanly.
- **Follow Existing Patterns**: Before writing new code, read neighboring files in the same package to match established conventions (arena usage, `runtime.AddCleanup` / `runtime.KeepAlive`, `verifyAndCastValues`, closure caching, etc.).
- **Avoid Large Files (Split Per-Op Implementations > 200 Lines)**: Keep files focused so directory listings stay scannable. Place complex op families (> ~200 lines, e.g., `ops_dotgeneral.go`, `ops_reduce.go`, `ops_scatter.go`, `flash.go`) in dedicated files rather than growing a monolith.
- **Copyright Notes**: Normal source files start with:
  ```go
  // Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0
  ```
  Auto-generated files (`gen_*`) omit the copyright header and instead carry a comment identifying the generator tool.

### Compliance Tests, Unit Tests & Benchmarks

- Generic backend tests and benchmarks belong in `github.com/gomlx/compute/support/backendtest` (`TestCompliance` and `BenchmarkCompliance` in `compute/xla/xla_test.go` invoke `backendtest.RunAll` and `backendtest.RunAllBenchmarks`).
- Common test commands:
  1. Pure-Go StableHLO & shape inference tests (fast, no PJRT plugin required):
     ```bash
     go test ./stablehlo/... ./types/... ./internal/shapeinference/...
     ```
  2. `compute/xla` backend tests on CPU or CUDA:
     ```bash
     GOMLX_BACKEND=xla:cpu go test ./compute/xla/...
     GOMLX_BACKEND=xla:cuda go test -p 1 ./compute/xla/...
     ```
  3. Low-level `pjrt` and end-to-end integration tests:
     ```bash
     go test -p 1 ./pjrt/... ./internal/tests/...
     ```
  4. Standard backend benchmarks:
     ```bash
     GOMLX_BACKEND=xla:cpu go test -run none -bench BenchmarkCompliance ./compute/xla
     ```
- **Always pass `-p 1` when testing multiple packages with a GPU**: Direct `pjrt` CUDA clients preallocate ~75% of GPU memory by default, so concurrent package test binaries will fail with OOM if run in parallel.

### Code Generation (`go generate`)

- Standalone generators (no external C++ repo required):
  - `compute/xla`: `go generate ./compute/xla` runs `internal/cmd/computexla_generator` to generate `gen_unary_ops.go` and `gen_binary_ops.go` (pass `-op=<OpName>` to print a skeleton for a new manual op).
  - `stablehlo`: `go generate ./stablehlo` runs `internal/cmd/ops_generator` to generate `gen_unary_ops.go` and `gen_binary_ops.go`.
  - `types` & `internal/optypes`: uses `go tool enumer` for `gen_*_enumer.go`.
  - `pjrt`: `internal/cmd/copy_go_code` copies `internal/chelper.go` into `pjrt/gen_chelper.go` (because CGo types cannot cross Go package boundaries).
- OpenXLA-dependent generators (require `XLA_SRC` pointing to a local clone of `github.com/openxla/xla` plus `protoc` / `protoc-gen-go`):
  - `internal/cmd/pjrt_codegen` (parses `pjrt_c_api.h` -> `pjrt/gen_api_calls.{c,h}`, `pjrt/gen_new_struct.{c,h}`, `pjrt/gen_c_enums.go`).
  - `internal/cmd/protoc_xla_protos` (regenerates Go protobuf bindings under `internal/protos/`).

---

## Package Directory Structure

```text
github.com/gomlx/go-xla/
├── compute/xla/            # compute.Backend implementation ("xla", "stablehlo", "shlo", "hlo")
│   ├── xla.go              # Backend registration, config/option parsing, plugin lookup, AutoInstaller hook
│   ├── backend.go          # Backend & Buffer implementations (DeviceNum, ToFlatData, SharedBuffers)
│   ├── builder.go          # Builder, Node, binary broadcasting, SPMD & Shardy auto-sharding setup
│   ├── function.go         # Function, Parameter, Constant, Return, Closures, Sort, While, If
│   ├── executable.go       # Compile() (Builder -> StableHLO -> PJRT) and Execute()
│   ├── capabilities.go     # Supported compute.OpType and dtypes.DType matrix
│   ├── ops.go              # Core op lowerings, Call, OptimizationBarrier, SchedulingBarrier, customCall
│   ├── ops_dotgeneral.go   # DotGeneral lowering, TF32 config, CPU weight-conversion barrier
│   ├── ops_reduce.go       # Reduce*, ReduceWindow, ArgMinMax (NaN + tie-breaking), SelectAndScatter*
│   ├── ops_scatter.go      # ScatterSum, ScatterMax, ScatterMin
│   ├── ops_collective.go   # AllReduce (splits mixed-DType operands per StableHLO rules)
│   ├── fused_ops.go        # ErrNotImplemented stubs for unfused compute.FusedOps
│   ├── flash.go            # CUDA cuDNN Flash Attention (FusedScaledDotProductAttention + VJP)
│   ├── gen_*_ops.go        # Auto-generated unary/binary compute.Function wrappers
│   ├── autoinstall/        # Side-effect import wiring installer.AutoInstall into compute/xla
│   └── cpu/{static,dynamic}/ # Optional build-time CGo pre-linking of CPU PJRT plugin
│
├── stablehlo/              # Pure-Go StableHLO MLIR text builder (no CGo)
│   ├── builder.go          # Module Builder, top-level functions, Shardy meshes, Write()/Build()
│   ├── function.go         # Function scope, NamedInput, Constant, Return, Closure management
│   ├── value.go            # SSA Value (%0, %arg0), shape & quantization attachment
│   ├── statement.go        # MLIR Statement formatting and tensor literal rendering
│   ├── ops.go              # Complex StableHLO ops (DotGeneral, Conv, Gather, Scatter, Dynamic*, etc.)
│   ├── computation.go      # Region-based ops (Reduce, ReduceWindow, Sort, While, If, Call)
│   ├── collective.go       # Collective ops (AllReduce, AllGather, CollectiveBroadcast, etc.)
│   ├── customcall.go       # stablehlo.custom_call / CustomCallV2 (layout & backend_config support)
│   └── gen_*_ops.go        # Auto-generated unary/binary StableHLO ops
│
├── pjrt/                   # Low-level CGo bindings to OpenXLA's pjrt_c_api.h
│   ├── pjrt_c_api.h        # Vendored OpenXLA PJRT C API header
│   ├── plugins.go          # Plugin singleton lifecycle, Attributes, NewClient, arena pool access
│   ├── dynamiclib_*.go     # OS-specific plugin search paths & dlopen/LoadLibrary wrappers
│   ├── clients.go          # PJRT_Client wrapper, device enumeration, compilation entry point
│   ├── compile.go          # CompileConfig builder (CompileOptionsProto, SPMD/Shardy, XLA_DEBUG_OPTIONS)
│   ├── loadedexecutables.go # LoadedExecutable, ExecutionConfig, C ExecuteAndWait, buffer donation
│   ├── executables.go      # Executable introspection (Name, NumOutputs, memory usage stats, HLO export)
│   ├── buffers*.go         # On-device Buffer lifecycle, synchronous H2D/D2H transfers, zero-copy CPU views
│   ├── arena.go            # Pooled C-allocated memory arenas for zero-malloc CGo argument marshaling
│   ├── alignedalloc.go     # 64-byte aligned C allocation (AlignedAlloc/AlignedFree) for shared CPU buffers
│   ├── cuda*.go            # Linux NVIDIA GPU detection (/dev/nvidia*, nvidia-smi) & XLA_FLAGS RPATH checks
│   ├── gen_*               # Auto-generated C trampolines, struct constructors, and enums
│   └── cpu/{static,dynamic}/ # Preloaded CPU PJRT plugin registration via C.GetPjrtApi()
│
├── types/                  # Shared XLA/StableHLO types
│   ├── ops.go              # Enumerations & configs (ComparisonType, DotGeneralAlgorithm, FFTType, etc.)
│   ├── shapes/             # go-xla Shape (with TupleShapes, DimensionBounds, and Quantization)
│   └── shardy/             # Shardy DeviceMesh, ShardingSpec, and replica-group computation
│
├── installer/              # Programmatic PJRT plugin downloader & cross-process flock installer
├── cmd/pjrt_installer/     # Standalone CLI installer (separate Go module with TUI dependencies)
└── internal/
    ├── shapeinference/     # Shape inference rules for `stablehlo` (kept separate from `compute/shapeinference`)
    ├── optypes/            # Internal StableHLO OpType enum
    ├── pool/               # Lock-free per-P intrusive object pool (`runtime_procPin`) used by CGo arenas
    ├── cbuffer/            # C-allocated byte buffer wrapper
    ├── protos/             # Generated Go protobufs for XLA `CompileOptionsProto`, `HloModuleProto`, etc.
    ├── rocm/               # AMD ROCm discrete GPU (`/dev/kfd` + `rocminfo`) and version detection
    ├── tests/              # End-to-end StableHLO + PJRT execution, Shardy, collective, and buffer tests
    └── cmd/                # Code generators (`pjrt_codegen`, `ops_generator`, `computexla_generator`, etc.)
```

---

## Architectural Separation & Execution Lifecycle

### 1. Graph Construction & Compilation (`stablehlo` & `compute/xla`)

- **Two `Shape` Packages**:
  - `github.com/gomlx/compute/shapes.Shape` is used at the `compute.Backend` boundary in `compute/xla`.
  - `github.com/gomlx/go-xla/types/shapes.Shape` is used inside `stablehlo`, `pjrt`, and `internal/shapeinference`. It diverged to support XLA `TupleShapes`, bounded dynamic dimensions (`DimUnknown = -1`, `DimensionBounds`, `EncodeBounds`), and StableHLO `Quantization` metadata. Both packages share `github.com/gomlx/compute/dtypes`, and `compute/xla` converts between them via `ShapeToXLA` and `ShapeFromXLA`.
- **Pure-Go StableHLO Emission (`stablehlo`)**:
  - `stablehlo` has **zero CGo dependencies**. `stablehlo.Builder` constructs a `module @name` containing a `main` `Function` (plus optional top-level functions for `Call` and nested `Closure`s for `Reduce`, `Scatter`, `Sort`, `While`, `If`, `AllReduce`).
  - Every op validates inputs and computes output shapes eagerly via `internal/shapeinference`.
  - **Scope & SSA Naming**: Statements automatically attach to the innermost `Function` scope among their operands (so operations using closure arguments are emitted inside that closure). Temporary value IDs (`%0`, `%1`, ...) and default argument IDs (`%arg0`, `%arg1`, ...) are allocated from counters on the **root** `Function` (`findRootFn()`) so closure SSA names never shadow or collide with parent function values.
  - `Builder.Build()` validates and serializes the AST into human-readable MLIR StableHLO text (`[]byte`), which PJRT's `"mlir"` program format accepts directly.
- **Compilation (`compute/xla/executable.go` & `pjrt/compile.go`)**:
  - `Builder.Compile()` renders the StableHLO bytes (logged when `klog.V(2)` is enabled) and invokes `client.Compile().WithStableHLO(program)`.
  - Single-device compilations without an explicit `DeviceAssignment` set `CompilePortableExecutable: true`, allowing the resulting `Executable` to run on any addressable `defaultDevice` passed to `Execute()`.
  - Multi-device compilations configure either `WithSPMD(numDevices)` or `WithShardy(numDevices)` + `WithDeviceAssignment(...)` inside `CompileOptionsProto`. If `XLA_DEBUG_OPTIONS` is set in the environment, `pjrt` parses it as textproto into `ExecutableBuildOptions.DebugOptions` (note: setting `XLA_DEBUG_OPTIONS` adds measurable XLA runtime overhead).

### 2. PJRT: Compilation and Execution Engine (`pjrt`)

- **Singleton `Plugin` & `Client`**:
  - `pjrt.GetPlugin(name)` dynamically loads the shared library (`dlopen` on Linux/macOS, `LoadLibrary` on Windows), resolves `GetPjrtApi`, calls `PJRT_Plugin_Initialize`, and caches the singleton `*Plugin`.
  - Alternatively, CPU plugins can be linked at build time (`RegisterPreloadedPlugin`) via build tags `pjrt_cpu_static` or `pjrt_cpu_dynamic` (or blank-importing `compute/xla/cpu/{static,dynamic}`).
- **Deterministic Cleanup + GC Safety (`runtime.AddCleanup`)**:
  - `Client`, `LoadedExecutable`, `Executable`, `Buffer`, and `arenaContainer` store raw C pointers in inner wrapper structs (`clientC`, `loadedExecutableC`, `bufferWrapper`, `arenaData`) registered with Go's `runtime.AddCleanup` so GC collection never leaks C/device resources, while `.Finalize()` / `.Destroy()` still release memory immediately (and stop pending cleanups on `Buffer.Destroy()`).
  - Go slices passed into C (`Compile`, `ToHost`, `BufferFromHost`) are pinned via `runtime.Pinner` and guarded with `runtime.KeepAlive`.
- **Single-CGo-Call Event Synchronization**:
  - PJRT execution and host transfers are asynchronous at the C API level and return a `PJRT_Event*`. To avoid multiple CGo transitions on the hot path, C helpers (`ExecuteAndWait` in `loadedexecutables.go`, `BufferFromHostAndWait` in `buffers_from_host.go`, and `BufferToHost` in `buffers_to_host.go`) invoke the PJRT call, `PJRT_Event_Await`, and `PJRT_Event_Destroy` within a single CGo call.
- **Distributed Execution**:
  - **SPMD**: Replicates one program across `numDevices`; `Executable.Execute` expects `numDevices * numParams` inputs in **device-major** order (`inputs[deviceIdx*numParams + paramIdx]`).
  - **AutoSharding (Shardy)**: Emits `sdy.mesh` headers on the module and `sdy.sharding` attributes on function inputs/outputs (`types/shardy`), enabling XLA's Shardy partitioner (`UseShardMapImport` / `UseAutoSpmdPartitioning`).

### 3. Device, Plugin Discovery & Auto-Installation

- **Plugin Search Order (`pjrt/dynamiclib_*.go`)**:
  1. `PJRT_PLUGIN_LIBRARY_PATH` (colon-separated directory list), if set.
  2. Otherwise: user-local install dir (`~/.local/lib/go-xla` on Linux, `~/Library/Application Support/go-xla` on macOS, `~/AppData/Local/go-xla` on Windows), `/usr/local/lib/go-xla` (plus legacy `.../gomlx/pjrt` paths), `LD_LIBRARY_PATH` / `DYLD_LIBRARY_PATH`, and `/etc/ld.so.conf`.
  3. Matches `pjrt_c_api_<name>_plugin.{so,dylib,dll}` and `pjrt[-_]plugin[-_]<name>.{so,dylib,dll}`. `compute/xla` also resolves unversioned names to versioned plugins (e.g., `"cpu"` -> `"cpu_v0.83.1"`), defaulting in preference order `DefaultPlugins = []string{"cuda", "rocm", "cpu"}`.
- **Hardware Presence Gating**:
  - **CUDA (`pjrt/cuda_linux.go`)**: Skips CUDA plugins during discovery unless `/dev/nvidia*` exists or `nvidia-smi` succeeds. On load, verifies that `../nvidia` exists next to the plugin directory (required by JAX wheel `$ORIGIN` `RPATH`s) and appends `--xla_gpu_cuda_data_dir=<nvidiaPath>` to `XLA_FLAGS` (set `GOPJRT_CUDA_CHECKS=0` to silence).
  - **ROCm (`internal/rocm/rocm.go`)**: Skips ROCm plugins unless `/dev/kfd` exists, `rocminfo` reports a discrete `gfx*` GPU (filtering out integrated APUs), and a ROCm `.info/version` file is found under `ROCM_PATH` / `/opt/rocm`.
- **Decoupled Auto-Installation (`compute/xla/autoinstall` & `installer`)**:
  - `compute/xla` exposes `AutoInstaller` without importing `installer` directly (keeping binary size and dependencies minimal when plugins are pre-installed). Blank-importing `_ "github.com/gomlx/go-xla/compute/xla/autoinstall"` registers automatic plugin downloads on `xla.New(...)` / `GetAvailablePlugins()` (disable via `GOMLX_NO_AUTO_INSTALL=1` or `xla.EnableAutoInstall(false)`).
  - `installer` uses `github.com/gofrs/flock` (`<plugin>.lock`) so concurrent processes never race while downloading/extracting plugins.
- **Backend Configuration Options (`xla.New(config)`)**:
  - Passed after the plugin name in `GOMLX_BACKEND="xla:<plugin>,<opt1>,<opt2>=<val>"` (use `"xla:help"` or `"xla:cpu,help"` to log options):
    - `tf32` / `notf32` (bool, default `true` on CUDA, `false` otherwise).
    - `shared_buffers` / `noshared_buffers` (bool, default `true` on CPU).
    - `preallocate` / `nopreallocate` (bool, default `false` on CUDA in `compute/xla`), `memory_fraction=<float>`, `allocator=<default|bfc|cuda_async|platform|vmm>`, `visible_devices=<0;1;...>`, `use_tfrt_gpu_client=<bool>`.
- **C++ Log Suppression**:
  - XLA plugins read `TF_CPP_MIN_LOG_LEVEL` **once** during `dlopen`. `compute/xla` sets `TF_CPP_MIN_LOG_LEVEL=3` lazily via `sync.OnceFunc` right before plugin loading (rather than in `init()`, avoiding `setenv` data races when linked as a C shared/archive library). Direct callers of `pjrt` should set `TF_CPP_MIN_LOG_LEVEL=3` before `pjrt.GetPlugin` or `pjrt.AvailablePlugins`.

---

## Nuanced Performance Details & Transfer Costs

### 1. GPU Zero-Copy Execution, Buffer Donation & Shared CPU Buffers

- **Device-Resident Execution**: On GPUs/TPUs, `xla.Buffer` wraps an on-device `*pjrt.Buffer`. Feeding outputs of one `Executable.Execute` into another keeps data completely on-device with zero PCIe traffic.
- **Host Transfers & Layout Nuance**:
  - `Buffer.Data()` uses `PJRT_Buffer_UnsafePointer`, which is only supported on CPU plugins; calling it on GPU/TPU buffers returns an error.
  - `Buffer.ToFlatData(dst)` performs a synchronous D2H copy (`PJRT_Buffer_ToHostBuffer`), explicitly requesting a dense major-to-minor host layout (`minor_to_major = [rank-1, ..., 0]`) because accelerators such as TPUs may store on-device buffers in tiled layouts.
- **Buffer Donation**: `Executable.Execute(inputs, donate, ...)` translates `donate` into PJRT `non_donatable_input_indices`. Donated input buffers can be aliased in-place for outputs by XLA (critical for zero-copy optimizer state/variable updates) and are automatically invalidated via `input.Destroy()` after execution.
- **Shared CPU Buffers (`HasSharedBuffers` / `NewSharedBuffer`)**: Enabled by default on `"cpu"`. Allocates 64-byte aligned C memory (`pjrt.AlignedAlloc` with `BufferAlignment = 64`) and wraps it via `PJRT_Client_CreateViewOfDeviceBuffer`, returning both the `compute.Buffer` and a direct Go slice over the same memory. **Caveat**: Shared buffers cannot be donated to `Execute()`, and must not be mutated while an execution reading them is in flight.

### 2. CGo Arena Pooling (`pjrt/arena.go` & `internal/pool`)

- Individual CGo calls and `C.malloc`/`C.free` calls add substantial latency on hot execution paths.
- Every `pjrt.Plugin` owns an `arenaPools` manager holding power-of-2 C-allocated arenas (from `2 KB` up to `16 MB`, providing bump-pointer 8-byte aligned sub-allocations for PJRT `*_Args` structs and pointer arrays).
- Rather than `sync.Pool` (which drops pooled C allocations every GC cycle), `arenaPools` uses `internal/pool.Pool`: a lock-free, 64-byte cache-line padded, per-P intrusive linked list (`runtime_procPin` / `runtime_procUnpin` via `//go:linkname`) whose pooled arenas never expire until `Plugin.Free()`.

### 3. Precision Conversions (`TF32`) & `DotGeneral` Lowering (`compute/xla/ops_dotgeneral.go`)

- **CUDA vs. Non-CUDA Lowering**:
  - On **CUDA**, `DotGeneral` configures TF32 via StableHLO `precision_config`: `DEFAULT` when `DotGeneralUseTF32 && accumulationDType == F32`, or explicitly `HIGHEST` when TF32 is disabled (leaving `precision_config` unset on CUDA silently permits TF32!).
  - On **non-CUDA plugins** (e.g., CPU), `DotGeneral` instead emits a `dot_algorithm` (`#stablehlo.dot_algorithm<...>`) specifying LHS, RHS, and accumulation precision types.
  - XLA only natively supports mixed-precision accumulation into `Float32`; if `config.AccumulatorDType` requests another type, `compute/xla` explicitly inserts `ConvertDType` on both operands first.
- **CPU Mixed-Precision Weight Memory Explosion Workaround (`dotGeneralAddDependency`)**:
  - On CPU PJRT, when `accumulationDType != inputDType` (e.g., `BFloat16` model weights accumulated in `Float32` across a 48-layer LLM), XLA's scheduler otherwise eagerly converts **all** layer weights to `Float32` in parallel at the start of the graph, multiplying peak RAM usage.
  - `ops_dotgeneral.go` injects a `SchedulingBarrier` making the weight parameter depend on the activation operand so XLA converts one layer's weights at a time.

### 4. Concurrency & Thread Locking

- **Graph Building**: `stablehlo.Builder` and `compute/xla.Builder` are **not** goroutine-safe; build each computation sequentially on a single goroutine.
- **Compilation & Execution**: Once compiled, `pjrt.LoadedExecutable` / `compute/xla.Executable` can be executed concurrently across goroutines. Each `Execute()` borrows its own C `arenaContainer` from the per-P pool and awaits its own `PJRT_Event`s. Unlike ONNX Runtime, PJRT does not require `runtime.LockOSThread()`.
- **Plugin Loading & Installation**: `pjrt.GetPlugin` is serialized with `muPlugins`, and `installer.AutoInstall` is synchronized across OS processes with `flock`.

### 5. Dynamic Shapes

- **`stablehlo` Package**: Supports polymorphic/dynamic dimensions (`shapes.DimUnknown = -1`, formatted as `?` in MLIR), bounded dynamism (`DimensionBounds` + `EncodeBounds`, formatted as `#stablehlo.bounds<...>`), and dynamic ops (`DynamicReshape`, `DynamicBroadcastInDim`, `DynamicIota`, `DynamicGather`, `DynamicPad`, `DynamicConv`).
- **`compute/xla` Backend**: Advertises `DynamicShapes: compute.DynamicShapesNone` and rejects dynamic `Parameter` shapes. Because XLA/PJRT JIT-recompiles whenever concrete input shapes change rather than running shape-polymorphic kernels cheaply, callers (such as GoMLX) handle dynamic input sizes via power-of-2 bucketing/padding and per-shape `Executable` caching.

### 6. Notable Operator Lowering Nuances (`compute/xla`)

- **Fused Ops & CUDA cuDNN Flash Attention (`fused_ops.go`, `flash.go`)**:
  - Standard fused ops (`FusedSoftmax`, `FusedGelu`, `FusedLayerNorm`, `FusedDense`, `FusedAttentionQKVProjection`, `FusedQuantizedDense*`, `QuantizedEmbeddingLookup`) intentionally return `compute.ErrNotImplemented` because XLA's HLO fusion compiler fuses the decomposed primitives automatically.
  - **Exception — CUDA Flash Attention**: When `backend.IsCUDA()`, `Capabilities` enables `OpTypeFusedScaledDotProductAttention` and `OpTypeFusedScaledDotProductAttentionVJP`. `flash.go` lowers `Float16` and `BFloat16` attention (`BSHD` and `BHSD` layouts, GQA, causal/padding masks, custom scale, and additive bias) to XLA's internal `__cudnn$fmhaSoftmax*` / `__cudnn$fmhaScaleBiasSoftmax*` `stablehlo.custom_call` (API v2), matching JAX's cuDNN fMHA lowering. Unsupported combinations (non-CUDA plugins, `Float32`/`Float64` inputs, or simultaneous `Bias` + `SeqLens`) return `compute.ErrNotImplemented` so callers seamlessly fall back to decomposed attention.
- **`SchedulingBarrier` & CPU `OptimizationBarrier` Bug (`ops.go`)**:
  - StableHLO has no native `SchedulingBarrier` op, so `compute/xla` lowers `SchedulingBarrier(operand, deps...)` by slicing a scalar element from each dependency, multiplying by `OptimizationBarrier(0)`, summing, and adding the resulting zero to `operand`.
  - Single-operand `OptimizationBarrier(x)` on CPU PJRT is treated as a no-op because XLA's CPU compiler has a bug with 1-arg `stablehlo.optimization_barrier` that produces corrupted results or crashes.
- **Complex `Abs`/`Conj` & `IsNaN` (`ops.go`)**:
  - StableHLO does not define `stablehlo.abs` on complex tensors or a native `conj` op: `Abs` on complex inputs lowers to `Sqrt(Real(z)*Real(z) + Imag(z)*Imag(z))`, `Conj` lowers to `Complex(Real(z), Negate(Imag(z)))`, and `IsNaN(x)` lowers to `NotEqual(x, x)`.
- **Constant Sub-Byte `Bitcast` Restriction (`ops.go`)**:
  - `Bitcast` from a `Constant` node directly to a packed sub-byte `DType` (`Int4`, `Int2`, `Uint4`, `Uint2`) returns an explicit error working around OpenXLA compiler bug `openxla/xla#38964`.
- **Cached Reduction Closures & `ArgMinMax` Semantics (`ops_reduce.go`)**:
  - `Builder` caches reduction, selection, and `ArgMinMax` closures per `(dtype, opType)` to avoid emitting duplicate inline regions across large graphs.
  - `ArgMinMax` lowers to a 2-input `stablehlo.reduce` (`MultiReduce`) over `(Iota, x)` that explicitly propagates `NaN`s on float inputs and breaks ties toward the minimum index.
- **Mixed-DType `AllReduce` (`ops_collective.go`)**:
  - StableHLO requires all operands of a single `stablehlo.all_reduce` to have identical `DType`s; `Function.AllReduce` groups operands by `DType`, emits one `AllReduce` per group, and scatters results back to the caller's original operand order.
