//go:build !((linux && amd64) || pjrt_all)

package installer

// HasNvidiaGPU tries to guess if there is an actual Nvidia GPU installed (as opposed to only the drivers/PJRT
// file installed, but no actual hardware).
// It does that by checking for the presence of the device files in /dev/nvidia*.
//
// This is helpful to try to sort out the mess of path for nvidia libraries.
// Sadly, NVidia drivers are badly organized at multiple levels -- search to see how many questions there are related
// to where/how to install to CUDA libraries.
//
// To disable this check set GOPJRT_CUDA_CHECKS=no or GOPJRT_CUDA_CHECKS=0.
//
// On non-amd64 architectures, we assume there is no Nvidia GPU, and thi simply returns false.
func HasNvidiaGPU() bool { return false }
