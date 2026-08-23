//go:build (linux && amd64) || pjrt_all

package installer

import (
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"
)

func TestRocmSelectWheel(t *testing.T) {
	// Mirrors the actual directory listing of
	// https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2.4/ (relevant subset).
	hrefs := []string{
		"jax_rocm7_plugin-0.8.2%2Brocm7.2.4-cp311-cp311-manylinux_2_28_x86_64.whl",
		"jax_rocm7_pjrt-0.8.2%2Brocm7.2.4-py3-none-linux_x86_64.whl",
		"jax_rocm7_pjrt-0.8.2%2Brocm7.2.4-py3-none-manylinux2014_x86_64.whl",
		"jax_rocm7_pjrt-0.8.2%2Brocm7.2.4-py3-none-manylinux_2_28_x86_64.whl",
		"jaxlib-0.8.2%2Brocm7.2.4-cp311-cp311-manylinux_2_28_x86_64.whl",
	}

	got := rocmSelectWheel(hrefs)
	want := "jax_rocm7_pjrt-0.8.2%2Brocm7.2.4-py3-none-manylinux_2_28_x86_64.whl"
	if got != want {
		t.Fatalf("rocmSelectWheel() = %q, want %q", got, want)
	}
}

func TestRocmSelectWheelPrefersManylinux2014(t *testing.T) {
	// When manylinux_2_28 is absent, manylinux2014 should win over linux_x86_64.
	hrefs := []string{
		"jax_rocm7_pjrt-0.8.2%2Brocm7.2.4-py3-none-linux_x86_64.whl",
		"jax_rocm7_pjrt-0.8.2%2Brocm7.2.4-py3-none-manylinux2014_x86_64.whl",
	}
	got := rocmSelectWheel(hrefs)
	want := "jax_rocm7_pjrt-0.8.2%2Brocm7.2.4-py3-none-manylinux2014_x86_64.whl"
	if got != want {
		t.Fatalf("rocmSelectWheel() = %q, want %q", got, want)
	}
}

func TestRocmSelectWheelIgnoresNonPjrt(t *testing.T) {
	// The compiler plugin ("jax_rocm*_plugin") and other wheels must be ignored.
	hrefs := []string{
		"jax_rocm7_plugin-0.8.2%2Brocm7.2.4-cp311-cp311-manylinux_2_28_x86_64.whl",
		"jaxlib-0.8.2%2Brocm7.2.4-cp311-cp311-manylinux_2_28_x86_64.whl",
		"torch-2.7.1%2Brocm7.2.4-cp311-cp311-linux_x86_64.whl",
	}
	if got := rocmSelectWheel(hrefs); got != "" {
		t.Fatalf("rocmSelectWheel() = %q, want empty", got)
	}
}

func TestRocmWheelPriority(t *testing.T) {
	cases := []struct {
		name string
		want int
	}{
		{"manylinux_2_28_x86_64.whl", 3},
		{"manylinux2014_x86_64.whl", 2},
		{"manylinux1_x86_64.whl", 1},
		{"linux_x86_64.whl", 0},
	}
	for _, c := range cases {
		if got := rocmWheelPriority("jax_rocm7_pjrt-0.8.2-py3-none-" + c.name); got != c.want {
			t.Errorf("rocmWheelPriority(%q) = %d, want %d", c.name, got, c.want)
		}
	}
}

func TestFetchHTMLHrefs(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(`<html><body>
<a href="../">../</a>
<a href="jax_rocm7_pjrt-0.8.2%2Brocm7.2.4-py3-none-manylinux_2_28_x86_64.whl">foo</a>
<a href="torch-2.7.1%2Brocm7.2.4-cp311-cp311-linux_x86_64.whl">bar</a>
</body></html>`))
	}))
	defer server.Close()

	hrefs, err := fetchHTMLHrefs(server.URL + "/rocm-rel-7.2.4/")
	if err != nil {
		t.Fatalf("fetchHTMLHrefs failed: %v", err)
	}
	if len(hrefs) != 3 {
		t.Fatalf("fetchHTMLHrefs returned %d hrefs, want 3: %v", len(hrefs), hrefs)
	}
	if hrefs[0] != "../" {
		t.Fatalf("fetchHTMLHrefs[0] = %q, want %q", hrefs[0], "../")
	}
}

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
	if !rocminfoHasDiscreteGPU(discreteAndAPU) {
		t.Error("rocminfoHasDiscreteGPU() = false, want true (has a discrete GPU)")
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
	if rocminfoHasDiscreteGPU(apuOnly) {
		t.Error("rocminfoHasDiscreteGPU() = true, want false (APU only)")
	}
}

func TestRocminfoField(t *testing.T) {
	block := "  Name:                    gfx1100\n  Marketing Name:          AMD Radeon RX 7900 XTX\n"
	if got := rocminfoField(block, "Name:"); got != "gfx1100" {
		t.Errorf("rocminfoField(Name:) = %q, want %q", got, "gfx1100")
	}
	if got := rocminfoField(block, "Marketing Name:"); got != "AMD Radeon RX 7900 XTX" {
		t.Errorf("rocminfoField(Marketing Name:) = %q, want %q", got, "AMD Radeon RX 7900 XTX")
	}
	if got := rocminfoField(block, "Memory Properties:"); got != "" {
		t.Errorf("rocminfoField(Memory Properties:) = %q, want empty", got)
	}
}

func TestRocmInstallDirROCM_PATH(t *testing.T) {
	dir := t.TempDir()
	t.Setenv("ROCM_PATH", dir)
	t.Setenv("PATH", "") // Ensure rocminfo is not found in PATH.
	if got := rocmInstallDir(); got != dir {
		t.Fatalf("rocmInstallDir() = %q, want %q", got, dir)
	}
}

func TestRocmInstallDirFromPath(t *testing.T) {
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
	if got := rocmInstallDir(); got != root {
		t.Fatalf("rocmInstallDir() = %q, want %q", got, root)
	}
}

func TestRocmInstallDirDefault(t *testing.T) {
	t.Setenv("ROCM_PATH", "")
	t.Setenv("PATH", "")
	if got := rocmInstallDir(); got != "/opt/rocm" {
		t.Fatalf("rocmInstallDir() = %q, want /opt/rocm", got)
	}
}

func TestRocmDetectedVersion(t *testing.T) {
	root := t.TempDir()
	if err := os.MkdirAll(filepath.Join(root, ".info"), 0755); err != nil {
		t.Fatalf("failed to create .info dir: %v", err)
	}
	if err := os.WriteFile(filepath.Join(root, ".info", "version"), []byte("7.2.4\n"), 0644); err != nil {
		t.Fatalf("failed to write version file: %v", err)
	}
	t.Setenv("ROCM_PATH", root)
	t.Setenv("PATH", "")
	if got, err := RocmDetectedVersion(); err != nil || got != "7.2.4" {
		t.Fatalf("RocmDetectedVersion() = %q, %v; want %q, nil", got, err, "7.2.4")
	}
}

func TestRocmDetectedVersionNotFound(t *testing.T) {
	root := t.TempDir()
	t.Setenv("ROCM_PATH", root)
	t.Setenv("PATH", "")
	if got, err := RocmDetectedVersion(); err == nil || got != "" {
		t.Fatalf("RocmDetectedVersion() = %q, %v; want empty and error", got, err)
	}
}

func TestParseKFDProperty(t *testing.T) {
	content := `cpu_cores_count 0
simd_count 4
lds_size_in_kb 64
local_mem_size 0
max_engine_clk_ccompute 5756
`
	if val, ok := parseKFDProperty(content, "simd_count"); !ok || val != 4 {
		t.Errorf("parseKFDProperty(simd_count) = (%d, %v), want (4, true)", val, ok)
	}
	if val, ok := parseKFDProperty(content, "local_mem_size"); !ok || val != 0 {
		t.Errorf("parseKFDProperty(local_mem_size) = (%d, %v), want (0, true)", val, ok)
	}
	if val, ok := parseKFDProperty(content, "non_existent"); ok || val != 0 {
		t.Errorf("parseKFDProperty(non_existent) = (%d, %v), want (0, false)", val, ok)
	}
}

