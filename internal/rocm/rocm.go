//go:build (linux && amd64) || pjrt_all

package rocm

import (
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"

	"github.com/pkg/errors"
	"k8s.io/klog/v2"
)

// HasAMDGPU tries to guess if there is an actual discrete AMD GPU with ROCm
// installed. Integrated (APU) GPUs share system memory and are not suitable
// for ML workloads, so they are ignored.
//
// It checks for the presence of the /dev/kfd device file (the AMD ROCm compute
// device), verifies that ROCm drivers are properly installed (`rocminfo` is installed
// and the drivers with a proper version is found), and then uses `rocminfo` to
// determine if there is a discrete ROCm GPU.
var HasAMDGPU = sync.OnceValue(func() bool {
	if _, err := os.Stat("/dev/kfd"); err != nil {
		return false
	}
	output := RunRocminfo()
	if output == "" {
		// If there is no rocminfo installed.
		return false
	}
	if !RocminfoHasDiscreteGPU(output) {
		return false
	}
	// Verify that ROCm is actually installed (version file is present).
	if _, err := DetectedVersion(); err != nil {
		return false
	}
	return true
})

// InstallDir returns the ROCm installation directory. It is used to locate
// `rocminfo` and the ROCm version file when ROCm is not installed in the
// default /opt/rocm location.
//
// It checks, in order:
//  1. The ROCM_PATH environment variable, if set to an existing directory.
//  2. The install root inferred from the `rocminfo` binary found in PATH
//     (its parent directory's parent, e.g. <root>/bin/rocminfo -> <root>).
//  3. The default /opt/rocm.
func InstallDir() string {
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

// RunRocminfo executes `rocminfo` and returns its output, or "" if it is not
// available.
func RunRocminfo() string {
	var cmdPath string
	if p, err := exec.LookPath("rocminfo"); err == nil {
		cmdPath = p
	} else {
		candidate := filepath.Join(InstallDir(), "bin", "rocminfo")
		if _, err := os.Stat(candidate); err == nil {
			cmdPath = candidate
		} else {
			klog.V(1).Infof("rocminfo not found; assuming any AMD GPU is a discrete GPU")
			return ""
		}
	}
	cmd := exec.Command(cmdPath)
	output, err := cmd.CombinedOutput()
	if err != nil {
		klog.V(1).Infof("rocminfo failed to execute: %v", err)
		return ""
	}
	return string(output)
}

// RocminfoHasDiscreteGPU parses `rocminfo` output and reports whether at least
// one AMD GPU agent is a discrete GPU (as opposed to an integrated APU, which
// reports "Memory Properties: APU").
func RocminfoHasDiscreteGPU(output string) bool {
	blocks := strings.SplitSeq(output, "*******")
	for block := range blocks {
		name := RocminfoField(block, "Name:")
		if !strings.HasPrefix(name, "gfx") {
			continue // Not a GPU agent (e.g. the CPU agent).
		}
		if RocminfoField(block, "Memory Properties:") != "APU" {
			return true
		}
	}
	return false
}

// RocminfoField returns the value of the given field (e.g. "Name:") within a
// rocminfo agent block, or "" if not found.
func RocminfoField(block, field string) string {
	for line := range strings.SplitSeq(block, "\n") {
		line = strings.TrimSpace(line)
		if !strings.HasPrefix(line, field) {
			continue
		}
		return strings.TrimSpace(line[len(field):])
	}
	return ""
}

// DetectedVersion returns the installed ROCm version (e.g. "7.2.4"), read
// from the ROCm version file, located using InstallDir.
func DetectedVersion() (string, error) {
	root := InstallDir()
	for _, p := range []string{
		filepath.Join(root, ".info", "version"),
		filepath.Join(root, "lib", ".info", "version"),
	} {
		b, err := os.ReadFile(p)
		if err != nil {
			continue
		}
		version := strings.TrimSpace(string(b))
		if version != "" {
			return version, nil
		}
	}
	return "", errors.Errorf("could not detect the installed ROCm version: "+
		"no version file found under %q", root)
}
