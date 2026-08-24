//go:build !((linux && amd64) || pjrt_all)

package installer

// HasAMDGPU tries to guess if there is an actual AMD GPU with ROCm installed.
// On non-linux/amd64 platforms, ROCm is not supported, so this simply returns
// false.
func HasAMDGPU() bool { return false }
