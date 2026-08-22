//go:build linux && amd64

package pjrt

import (
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"

	"k8s.io/klog/v2"
)

// hasAMDGPU tries to guess if there is an actual discrete AMD GPU with ROCm
// installed. Integrated (APU) GPUs share system memory and are not suitable
// for ML workloads, so they are ignored.
//
// It checks for the presence of the /dev/kfd device file (the AMD ROCm compute
// device) and then parses `rocminfo` to tell a discrete GPU apart from an APU.
var hasAMDGPU = sync.OnceValue[bool](func() bool {
	if _, err := os.Stat("/dev/kfd"); err != nil {
		return false
	}
	output := runRocminfo()
	if output == "" {
		// Can't run rocminfo; assume the AMD GPU is usable.
		return true
	}
	return rocminfoHasDiscreteGPU(output)
})

// rocmInstallDir returns the ROCm installation directory. It is used to locate
// `rocminfo` when ROCm is not installed in the default /opt/rocm location.
//
// It checks, in order:
//  1. The ROCM_PATH environment variable, if set to an existing directory.
//  2. The install root inferred from the `rocminfo` binary found in PATH
//     (its parent directory's parent, e.g. <root>/bin/rocminfo -> <root>).
//  3. The default /opt/rocm.
func rocmInstallDir() string {
	if dir := os.Getenv("ROCM_PATH"); dir != "" {
		if info, err := os.Stat(dir); err == nil && info.IsDir() {
			return dir
		}
	}
	if p, err := exec.LookPath("rocminfo"); err == nil {
		if real, err := filepath.EvalSymlinks(p); err == nil {
			p = real
		}
		return filepath.Dir(filepath.Dir(p))
	}
	return "/opt/rocm"
}

// runRocminfo executes `rocminfo` and returns its output, or "" if it is not
// available.
func runRocminfo() string {
	var path string
	if p, err := exec.LookPath("rocminfo"); err == nil {
		path = p
	} else {
		candidate := filepath.Join(rocmInstallDir(), "bin", "rocminfo")
		if _, err := os.Stat(candidate); err == nil {
			path = candidate
		} else {
			klog.V(1).Infof("rocminfo not found; assuming any AMD GPU is a discrete GPU")
			return ""
		}
	}
	cmd := exec.Command(path)
	output, err := cmd.CombinedOutput()
	if err != nil {
		klog.V(1).Infof("rocminfo failed to execute: %v", err)
		return ""
	}
	return string(output)
}

// rocminfoHasDiscreteGPU parses `rocminfo` output and reports whether at least
// one AMD GPU agent is a discrete GPU (as opposed to an integrated APU, which
// reports "Memory Properties: APU").
func rocminfoHasDiscreteGPU(output string) bool {
	blocks := strings.Split(output, "*******")
	for _, block := range blocks {
		name := rocminfoField(block, "Name:")
		if !strings.HasPrefix(name, "gfx") {
			continue // Not a GPU agent (e.g. the CPU agent).
		}
		if rocminfoField(block, "Memory Properties:") != "APU" {
			return true
		}
	}
	return false
}

// rocminfoField returns the value of the given field (e.g. "Name:") within a
// rocminfo agent block, or "" if not found.
func rocminfoField(block, field string) string {
	idx := strings.Index(block, field)
	if idx < 0 {
		return ""
	}
	rest := block[idx+len(field):]
	if nl := strings.IndexByte(rest, '\n'); nl >= 0 {
		rest = rest[:nl]
	}
	return strings.TrimSpace(rest)
}
