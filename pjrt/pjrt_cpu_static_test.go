//go:build pjrt_cpu_static

package pjrt

import (
	"github.com/gomlx/go-xla/internal/pjrt/cpustatictest"
	"k8s.io/klog/v2"
)

// Duplicated from pjrt/cpu/static to avoid cyclic dependencies.
func init() {
	pjrtAPI := cpustatictest.GetPjrtApi()
	if pjrtAPI == 0 {
		klog.Fatal("Failed to get PJRT API pointer when initializing static PJRT (github.com/gomlx/go-xla/pkg/pjrt/static).")
	}
	err := RegisterPreloadedPlugin("cpu", pjrtAPI)
	if err != nil {
		klog.Fatalf("Failed to register static PJRT plugin for CPU (github.com/gomlx/go-xla/pkg/pjrt/static): %+v", err)
	}
}
