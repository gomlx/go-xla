// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

// Package xla implements a GoMLX backend, that is, a `github.com/gomlx/compute.Backend` interface using
// [Google's XLA (PJRT)](https://openxla.org/) as a backend.
//
// The backend is registered with the aliases "xla", "stablehlo", "shlo" or "hlo" (all aliases to the same backend).
//
// XLA/PJRT uses C++ written PJRT "plugins", `.so` files that implements XLA(PRJT). They are loaded dynamically in the program.
//
// By default, the this XLA backend loads the requested plugins after the program starts and specifies the desired
// plugin name (default to "cpu") using `dlopen`.
//
// If the plugins are not available, the backend will download them automatically ("auto-install):
//
// - From github.com/gomlx/pjrt-cpu-binaries for CPU PJRT plugins.
// - From pypi.org, using the Jax pacakges for the CUDA and TPU PJRT plugins.
//
// Auto-install has no effect if default plugins are already installed. But to control it you can:
//
//   - Call xla.AutoInstall() directly if you want to call it immediately.
//   - Configure it with xla.EnableAutoInstall() if you want to enable/disable it globally (default is enabled).
//   - Set GOMLX_NO_AUTO_INSTALL, which sets the global auto-install flag to false -- but it can be overridden by
//     calling xla.EnableAutoInstall().
//
// Experimentally, one can get this backend to work with pre-linked PJRT plugins, but it will require the user to
// add the `.so` files in a library in LD_LIBRARY_PATH, or precompile a `.a` static library.
//
//   - Pre-link the CPU PJRT plugin statically: this will generate a bigger binary (+ ~200Mb, so slower to build),
//     but allows one to build a static binary that can be deployed without extra dependencies (except the standard C and C++ libraries,
//     usually available in most machines).
//     To enable, build using the tag `pjrt_cpu_static` (e.g.: `go build --tags pjrt_cpu_static ...`),
//     or import `github.com/gomlx/gomlx/backends/xla/cpu/static`. Both methods have the same effect.
//   - Pre-link the CPU PJRT plugin dynamically: build with the build tag `pjrt_cpu_dynamic` (e.g.: `go test --tags pjrt_cpu_dynamic ...`),
//     or import `github.com/gomlx/gomlx/backends/xla/cpu/dynamic`. Not much difference from linking the PJRT plugin
//     after the program starts, as default.
//
// # Shared Buffers Support:
//
// XLA/PJRT for CPU allows the "device buffer" (where device=CPU) to be addressed directly, which
// saves the copy from "host/local tensor" to the "on-device tensor" when executing a computation.
// This is enabled by default if the plugin is called "cpu". To force advertising support for this
// for other PJRTs provide the "shared_buffers" option, e.g.: GOMLX_BACKEND="xla:my_pjrt,shared_buffers".
// Or to force disabling the support, provide the "noshared_buffers" option.
//
// # Options
//
// Those can be passed after the plugin name, e.g.: GOMLX_BACKEND="xla:cuda,tf32=false,preallocate=false".
//
//   - "tf32" (boolean, default=true): controls whether to use TF32 for DotGeneral operations that are using float32
//     (it can be faster in modern GPUs). It's enabled by default.
//   - "shared_buffer" (boolean, default=true): controls whether to use shared buffers for the device buffer
//     (where device=CPU). It's enabled by default if the plugin is called "cpu".
//   - "preallocate" (boolean, default=true): whether the CUDA PJRT preallocates a large portion of the memory.
//   - "memory_fraction" (float, default=0.75): how much memory to preallocate.
//   - "allocator" (string, default="default"): which allocator to use. For CUDA the available ones are "default"
//     (== "bfc"), "bfc" ("best-fit for coalescing", avoids framementation), "cuda_async" (dynamic, no preallocation),
//     "platform" (slow, good for debugging), "vmm"
//   - "visible_devices" (list of integers, e.g., "0;1;2"): list IDs of the devices made visible to the backend.
//   - "use_tfrt_gpu_client" (boolean, default=false): uses the "TFRT" dispatcher for GPU.
//
// # (NO) Dynamic Shapes
//
// XLA doesn't support dynamic shapes. Sort of ... it suppots, but any new shape triggers a re-compilation, something
// that GoMLX and other [compute.Backend] clients already supprot.
//
// So it does NOT support dynamic shapes in the [compute.Backend] sense, which implies a low cost execution overhead
// for various shapes.
package xla

import (
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"strings"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/shapes"
	"github.com/gomlx/compute/support/sets"
	"github.com/gomlx/compute/support/xslices"
	"github.com/gomlx/go-xla/installer"
	"github.com/gomlx/go-xla/pjrt"
	xlashapes "github.com/gomlx/go-xla/types/shapes"
	"github.com/pkg/errors"
	"k8s.io/klog/v2"
)

//go:generate go run ../../internal/cmd/computexla_generator

// BackendName is the name of the backend.
//
// The stablehlo backend also accepts the "xla", "hlo" and "pjrt" aliases.
const BackendName = "xla"

// Disable XLA logging by default by setting TF_CPP_MIN_LOG_LEVEL to 3 (above errors level), if it is not already set.
// This won't work if the PJRT is linked statically or dynamically before the go program start (without `dlopen` that is).
//
// See issue https://github.com/openxla/xla/issues/26466
func init() {
	const TensorflowCPPMinLogLevelEnv = "TF_CPP_MIN_LOG_LEVEL"
	tfLogLevel := os.Getenv(TensorflowCPPMinLogLevelEnv)
	if tfLogLevel == "" {
		err := os.Setenv(TensorflowCPPMinLogLevelEnv, "3")
		if err != nil {
			klog.Errorf("Failed to set $%s to 3: %v", TensorflowCPPMinLogLevelEnv, err)
		}
	}
}

// New returns a new Backend using the config as a configuration.
// The config string should be the name of the PJRT plugin to use.
//
// This function triggers AutoInstall if it is enabled (the default). See EnableAutoInstall to disable it.
func New(config string) (compute.Backend, error) {
	return NewWithOptions(config, nil)
}

// optionsDocumentation contains the help text for the options.
const optionsDocumentation = `"xla" backend extra options:
  - "tf32" (boolean, default=true): controls whether to use TF32 for DotGeneral operations that are using float32
    (it can be faster in modern GPUs). It's enabled by default.
  - "shared_buffer" (boolean, default=true): controls whether to use shared buffers for the device buffer
    (where device=CPU). It's enabled by default if the plugin is called "cpu".
  - "preallocate" (boolean, default=true): whether the CUDA PJRT preallocates a large portion of the memory.
  - "memory_fraction" (float, default=0.75): how much memory to preallocate.
  - "allocator" (string, default="default"): which allocator to use. For CUDA the available ones are "default"
    (== "bfc"), "bfc" ("best-fit for coalescing", avoids framementation), "cuda_async" (dynamic, no preallocation),
    "platform" (slow, good for debugging), "vmm"
  - "visible_devices" (list of integers, e.g., "0;1;2"): list IDs of the devices made visible to the backend.
  - "use_tfrt_gpu_client" (boolean, default=false): uses the "TFRT" dispatcher for GPU.`

// NewWithOptions creates a StableHLO backend with the given client options.
// It allows more control, not available with the default New constructor.
//
// This function triggers AutoInstall if it is enabled (the default). See EnableAutoInstall to disable it.
func NewWithOptions(config string, options pjrt.NamedValuesMap) (*Backend, error) {
	pluginName := config

	// Make shallow copy options, since we may change it:
	pluginOptions := make(pjrt.NamedValuesMap)
	for key, value := range options {
		pluginOptions[key] = value
	}

	// Parse backendOptions from config string.
	backendOptions := make(map[string]string)
	parts := strings.Split(config, ",")
	if len(parts) > 1 {
		pluginName = parts[0]
		for _, part := range parts[1:] {
			if part == "" {
				continue
			}
			key, val, found := strings.Cut(part, "=")
			if !found {
				backendOptions[part] = ""
			} else {
				backendOptions[key] = val
			}
		}
	}

	_, helpOptionSet := backendOptions["help"]
	if pluginName == "help" || helpOptionSet {
		klog.Infof("Available plugins: %q", GetAvailablePlugins())
		klog.Info(optionsDocumentation)
		return nil, errors.New("Help requested")
	}

	// FInd plugin.
	if !filepath.IsAbs(pluginName) {
		if autoInstall {
			err := AutoInstall()
			if err != nil {
				return nil, errors.WithMessagef(err, "backend %q failed to auto-install default plugins", BackendName)
			}
		}

		// Verify the pluginName is available.
		plugins := GetAvailablePlugins()
		if len(plugins) == 0 {
			return nil, errors.Errorf("no plugins found for backend %q -- either use the absolute "+
				"path to the pluginName as the configuration or set PJRT_PLUGIN_LIBRARY_PATH to the path where to search for "+
				"PJRT plugins", BackendName)
		}
		if pluginName == "" {
			pluginName = plugins[0]
		} else if slices.Index(plugins, pluginName) == -1 {
			// Try to find a versioned plugin matching the base name (e.g., "cpu" matches "cpu_v0.83.1")
			versionedName := findVersionedPlugin(pluginName, plugins)
			if versionedName == "" {
				return nil, errors.Errorf("Plugin %q for backend %q not found: available plugins found %q", pluginName, BackendName, plugins)
			}
			pluginName = versionedName
		}
	}

	// Create backend option (not associated with a plugin yet)
	backend := &Backend{
		pluginName:   pluginName,
		config:       config,
		capabilities: Capabilities.Clone(),

		// Enable TF32 by default for CUDA.
		DotGeneralUseTF32: isPluginType(pluginName, "cuda"),

		// SharedBuffers is true for CPU by default
		hasSharedBuffers: isPluginType(pluginName, "cpu"),
	}

	// Support "shared buffers":
	if b, found, err := parseOptions[bool]("shared_buffers", backendOptions); err != nil {
		return nil, err
	} else if found {
		backend.hasSharedBuffers = b
	}

	// Support for tf32 DotGeneral.
	if b, found, err := parseOptions[bool]("tf32", backendOptions); err != nil {
		return nil, err
	} else if found {
		backend.DotGeneralUseTF32 = b
	}

	// Support "preallocate":
	if b, found, err := parseOptions[bool]("preallocate", backendOptions); err != nil {
		return nil, err
	} else if found {
		pluginOptions["preallocate"] = b
	}

	// Control memory fraction of preallocated memory:
	if f, found, err := parseOptions[float32]("memory_fraction", backendOptions); err != nil {
		return nil, err
	} else if found {
		pluginOptions["memory_fraction"] = f
	}

	// Allocator to use for CUDA:
	if s, found, err := parseOptions[string]("allocator", backendOptions); err != nil {
		return nil, err
	} else if found {
		pluginOptions["allocator"] = s
	}

	// Visible devices for the client:
	if visibleDevices, found, err := parseOptions[[]int64]("visible_devices", backendOptions); err != nil {
		return nil, err
	} else if found {
		pluginOptions["visible_devices"] = visibleDevices
	}

	// Use TFRT GPU client:
	if useTFRT, found, err := parseOptions[bool]("use_tfrt_gpu_client", backendOptions); err != nil {
		return nil, err
	} else if found {
		pluginOptions["use_tfrt_gpu_client"] = useTFRT
	}

	// Any leftover plugin options are unknown.
	if len(backendOptions) != 0 {
		klog.Errorf("backend %q: unknown plugin options %v", BackendName, xslices.SortedKeys(backendOptions))
	}

	// Create plugin.
	plugin, err := pjrt.GetPlugin(pluginName)
	if err != nil {
		return nil, errors.WithMessagef(err, "backend %q:", BackendName)
	}
	var client *pjrt.Client
	client, err = plugin.NewClient(pluginOptions)
	if err != nil {
		return nil, errors.WithMessagef(err, "while creating plugin %s for backend %q", pluginName, BackendName)
	}
	klog.V(1).Infof("created new plugin %q for backend %q", pluginName, BackendName)

	// Set backend.
	backend.plugin = plugin
	backend.client = client
	backend.numDevices = len(client.AddressableDevices())

	return backend, nil
}

// parseOptions parses the optionName from backendOptions (string).
// If optionName is found, it's removed from backendOptions.
// For bool options, it also searches for "no"+optionName, and if found, removes it and returns false.
// It returns the parsed value, whether it was found, and any parsing error.
func parseOptions[T interface {
	string | bool | float32 | []int64
}](
	optionName string, backendOptions map[string]string) (T, bool, error) {
	var val T

	if _, ok := any(val).(bool); ok {
		noKey := "no" + optionName
		if _, foundNo := backendOptions[noKey]; foundNo {
			delete(backendOptions, noKey)
			return any(false).(T), true, nil
		}
	}

	valStr, found := backendOptions[optionName]
	if !found {
		return val, false, nil
	}
	delete(backendOptions, optionName)

	switch any(val).(type) {
	case string:
		return any(valStr).(T), true, nil
	case bool:
		if valStr == "" {
			return any(true).(T), true, nil
		}
		b, err := strconv.ParseBool(valStr)
		if err != nil {
			return val, true, errors.Wrapf(err, "Failed to parse option %q=%q", optionName, valStr)
		}
		return any(b).(T), true, nil
	case float32:
		f, err := strconv.ParseFloat(valStr, 32)
		if err != nil {
			return val, true, errors.Wrapf(err, "Failed to parse option %q=%q", optionName, valStr)
		}
		return any(float32(f)).(T), true, nil
	case []int64:
		if valStr == "" {
			return val, true, nil
		}
		parts := strings.FieldsFunc(valStr, func(r rune) bool {
			return r == ';' || r == ':' || r == ' '
		})
		res := make([]int64, 0, len(parts))
		for _, part := range parts {
			valInt, err := strconv.ParseInt(part, 10, 64)
			if err != nil {
				return val, true, errors.Wrapf(err, "Failed to parse option %q=%q", optionName, valStr)
			}
			res = append(res, valInt)
		}
		return any(res).(T), true, nil
	default:
		panic("unreachable")
	}
}

// Registers New() as the default constructor for "xla" backend.
func init() {
	compute.Register(BackendName, New)

	// Other aliases for this backend.
	compute.Register("stablehlo", New)
	compute.Register("hlo", New)
	compute.Register("shlo", New)
}

var (
	// DefaultPlugins is the list of plugins to use in preference order, if not otherwise specified.
	DefaultPlugins = []string{"cuda", "cpu"}

	// availablePluginsList are the keys to the available plugins sorted by DefaultPlugins.
	availablePluginsList []string
)

var autoInstall bool = true // Whether it should always auto-install at every call to New()

func init() {
	_, found := os.LookupEnv(NoAutoInstallEnv)
	if found {
		autoInstall = false
	}
}

const NoAutoInstallEnv = "GOMLX_NO_AUTO_INSTALL"

// AutoInstall the standard plugin version tested for the current go-xla version.
// If GPU or TPU are detected, it will also install the corresponding plugins.
//
// This simply calls github.com/gomlx/go-xla/installer.AutoInstall().
// If you want more control over the installation path, cache usage, or verbosity,
// you can use the AutoInstall function from go-xla's installer package directly.
func AutoInstall() error {
	return installer.AutoInstall("", true, installer.Normal)
}

// EnableAutoInstall sets whether AutoInstall should be triggered automatically for GetAvailablePlugins or New.
//
// If enabled, the default, the AutoInstall function will be called automatically when GetAvailablePlugins or New is called.
func EnableAutoInstall(enable bool) {
	autoInstall = enable
}

// GetAvailablePlugins lists the available platforms -- it caches and reuses the result in future calls.
//
// This function triggers AutoInstall if it is enabled (the default). See EnableAutoInstall to disable it.
//
// Plugins are searched in the PJRT_PLUGIN_LIBRARY_PATH directory -- or directories if it is a ":" separated list.
// If it is not set, it will search the system "/usr/local/lib/go-xla", the users $HOME/.local/lib/go-xla (or
// "$HOME/Library/Application Support/go-xla" in MacOS) and the standard libraries directories of the
// system (in linux in LD_LIBRARY_PATH and /etc/ld.so.conf file) in that order.
//
// If there are plugins with the same name but different versions in different directories, it respects the order
// of the directories given by PJRT_PLUGIN_LIBRARY_PATH or by the system.
//
// See details in pjrt.AvailablePlugins.
func GetAvailablePlugins() []string {
	if autoInstall {
		err := AutoInstall()
		if err != nil {
			klog.Errorf("Error auto-installing plugins: %+v", err)
		}
	}

	if len(availablePluginsList) > 0 {
		// Use cache results.
		return availablePluginsList
	}

	availablePluginsMap := pjrt.AvailablePlugins()
	pluginNames := sets.MakeWith(xslices.Keys(availablePluginsMap)...)
	klog.V(1).Infof("Available plugins: %v\n", pluginNames)
	availablePluginsList = make([]string, 0, len(pluginNames))

	// Add DefaultPlugins first.
	for _, pluginName := range DefaultPlugins {
		if pluginNames.Has(pluginName) {
			availablePluginsList = append(availablePluginsList, pluginName)
			delete(pluginNames, pluginName)
		}
	}

	// Add the other plugins in some random order.
	for pluginName := range pluginNames {
		availablePluginsList = append(availablePluginsList, pluginName)
	}
	return availablePluginsList
}

// findVersionedPlugin looks for a versioned plugin matching the given base name.
// For example, if baseName is "cpu" and plugins contains "cpu_v0.83.1", it returns "cpu_v0.83.1".
// If no versioned match is found, it returns an empty string.
func findVersionedPlugin(baseName string, plugins []string) string {
	prefix := baseName + "_v"
	for _, p := range plugins {
		if strings.HasPrefix(p, prefix) {
			return p
		}
	}
	return ""
}

// isPluginType checks if pluginName matches the given base type.
// For example, isPluginType("cpu_v0.83.1", "cpu") returns true.
func isPluginType(pluginName, baseType string) bool {
	return pluginName == baseType || strings.HasPrefix(pluginName, baseType+"_v")
}

// ShapeToXLA converts a GoMLX shape to a go-xla shape.
func ShapeToXLA(shape shapes.Shape) xlashapes.Shape {
	if !shape.Ok() || shape.IsTuple() {
		return xlashapes.Invalid()
	}
	return xlashapes.Make(shape.DType, slices.Clone(shape.Dimensions)...)
}

// ShapeFromXLA converts a go-xla shape to a GoMLX shape.
func ShapeFromXLA(shape xlashapes.Shape) shapes.Shape {
	if !shape.Ok() || shape.IsTuple() {
		return shapes.Invalid()
	}
	return shapes.Make(shape.DType, slices.Clone(shape.Dimensions)...)
}
