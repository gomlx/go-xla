//go:build (linux && amd64) || pjrt_all

package installer

import (
	"net/http"
	"net/http/httptest"
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

