//go:build !(linux && amd64)

package pjrt

// hasAMDGPU reports whether there is a discrete AMD GPU with ROCm installed.
// On non-linux/amd64 platforms, ROCm is not supported, so this simply returns
// false.
func hasAMDGPU() bool { return false }
