//go:build (linux && amd64) || pjrt_all

package rocm_test

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/gomlx/go-xla/internal/rocm"
)

func TestRocminfoHasDiscreteGPU(t *testing.T) {
	// A trimmed rocminfo output mirroring a host with a discrete card plus an APU.
	discreteAndAPU := `*******
Agent 1                  
*******
  Name:                    AMD Ryzen 9 7900X 12-Core Processor
  Marketing Name:          AMD Ryzen 9 7900X 12-Core Processor
  Vendor Name:             CPU
*******
Agent 2                  
*******
  Name:                    gfx1100                            
  Marketing Name:          AMD Radeon RX 7900 XTX             
  Vendor Name:             AMD                                
  Memory Properties:       
*******
Agent 3                  
*******
  Name:                    gfx1036                            
  Marketing Name:          AMD Ryzen 9 7900X 12-Core Processor
  Vendor Name:             AMD                                
  Memory Properties:       APU
`
	if !rocm.RocminfoHasDiscreteGPU(discreteAndAPU) {
		t.Error("rocm.RocminfoHasDiscreteGPU() = false, want true (has a discrete GPU)")
	}
}

func TestRocminfoHasDiscreteGPUAPUOnly(t *testing.T) {
	apuOnly := `*******
Agent 1                  
*******
  Name:                    gfx1036                            
  Marketing Name:          AMD Ryzen 9 7900X 12-Core Processor
  Vendor Name:             AMD                                
  Memory Properties:       APU
`
	if rocm.RocminfoHasDiscreteGPU(apuOnly) {
		t.Error("rocm.RocminfoHasDiscreteGPU() = true, want false (APU only)")
	}
}

func TestRocminfoField(t *testing.T) {
	block := "  Name:                    gfx1100\n  Marketing Name:          AMD Radeon RX 7900 XTX\n"
	if got := rocm.RocminfoField(block, "Name:"); got != "gfx1100" {
		t.Errorf("rocm.RocminfoField(Name:) = %q, want %q", got, "gfx1100")
	}
	if got := rocm.RocminfoField(block, "Marketing Name:"); got != "AMD Radeon RX 7900 XTX" {
		t.Errorf("rocm.RocminfoField(Marketing Name:) = %q, want %q", got, "AMD Radeon RX 7900 XTX")
	}
	if got := rocm.RocminfoField(block, "Memory Properties:"); got != "" {
		t.Errorf("rocm.RocminfoField(Memory Properties:) = %q, want empty", got)
	}
}

func TestInstallDirROCM_PATH(t *testing.T) {
	dir := t.TempDir()
	t.Setenv("ROCM_PATH", dir)
	t.Setenv("PATH", "") // Ensure rocminfo is not found in PATH.
	if got := rocm.InstallDir(); got != dir {
		t.Fatalf("rocm.InstallDir() = %q, want %q", got, dir)
	}
}

func TestInstallDirFromPath(t *testing.T) {
	root := t.TempDir()
	bin := filepath.Join(root, "bin")
	if err := os.MkdirAll(bin, 0755); err != nil {
		t.Fatalf("failed to create %q: %v", bin, err)
	}
	rocminfo := filepath.Join(bin, "rocminfo")
	if err := os.WriteFile(rocminfo, []byte("#!/bin/sh\n"), 0755); err != nil {
		t.Fatalf("failed to write %q: %v", rocminfo, err)
	}
	t.Setenv("ROCM_PATH", "")
	t.Setenv("PATH", bin)
	if got := rocm.InstallDir(); got != root {
		t.Fatalf("rocm.InstallDir() = %q, want %q", got, root)
	}
}

func TestInstallDirDefault(t *testing.T) {
	t.Setenv("ROCM_PATH", "")
	t.Setenv("PATH", "")
	if got := rocm.InstallDir(); got != "/opt/rocm" {
		t.Fatalf("rocm.InstallDir() = %q, want /opt/rocm", got)
	}
}

func TestDetectedVersion(t *testing.T) {
	root := t.TempDir()
	if err := os.MkdirAll(filepath.Join(root, ".info"), 0755); err != nil {
		t.Fatalf("failed to create .info dir: %v", err)
	}
	if err := os.WriteFile(filepath.Join(root, ".info", "version"), []byte("7.2.4\n"), 0644); err != nil {
		t.Fatalf("failed to write version file: %v", err)
	}
	t.Setenv("ROCM_PATH", root)
	t.Setenv("PATH", "")
	if got, err := rocm.DetectedVersion(); err != nil || got != "7.2.4" {
		t.Fatalf("rocm.DetectedVersion() = %q, %v; want %q, nil", got, err, "7.2.4")
	}
}

func TestDetectedVersionNotFound(t *testing.T) {
	root := t.TempDir()
	t.Setenv("ROCM_PATH", root)
	t.Setenv("PATH", "")
	if got, err := rocm.DetectedVersion(); err == nil || got != "" {
		t.Fatalf("rocm.DetectedVersion() = %q, %v; want empty and error", got, err)
	}
}
