//go:build !((linux && amd64) || pjrt_all)

package rocm

import "github.com/pkg/errors"

// HasAMDGPU tries to guess if there is an actual AMD GPU with ROCm installed.
// On non-linux/amd64 platforms, ROCm is not supported, so this simply returns
// false.
func HasAMDGPU() bool { return false }

// InstallDir returns the ROCm installation directory.
// On non-linux/amd64 platforms, ROCm is not supported.
func InstallDir() string { return "/opt/rocm" }

// RunRocminfo executes `rocminfo` and returns its output.
// On non-linux/amd64 platforms, ROCm is not supported.
func RunRocminfo() string { return "" }

// RocminfoHasDiscreteGPU reports whether at least one AMD GPU agent is a discrete GPU.
// On non-linux/amd64 platforms, ROCm is not supported.
func RocminfoHasDiscreteGPU(output string) bool { return false }

// RocminfoField returns the value of the given field within a rocminfo agent block.
// On non-linux/amd64 platforms, ROCm is not supported.
func RocminfoField(block, field string) string { return "" }

// DetectedVersion returns the installed ROCm version.
// On non-linux/amd64 platforms, ROCm is not supported.
func DetectedVersion() (string, error) {
	return "", errors.New("ROCm is not supported on this platform")
}
