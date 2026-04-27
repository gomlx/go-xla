// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package xla_test

import (
	"fmt"
	"os"
	"testing"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/support/backendtest"
	"github.com/gomlx/compute/support/testutil"
	"github.com/gomlx/go-xla/compute/xla"
	"github.com/stretchr/testify/assert"
	"k8s.io/klog/v2"
)

func init() {
	klog.InitFlags(nil)
}

func testAllPlugins(t *testing.T, fn func(t *testing.T, backend compute.Backend, plugin string)) {
	envBackend := os.Getenv(compute.ConfigEnvVar)
	if envBackend != "" {
		backend, err := compute.New()
		if err != nil {
			t.Fatalf("Failed to create backend %q: %v", envBackend, err)
		}
		defer backend.Finalize()
		xlaBackend := backend.(*xla.Backend)
		fn(t, backend, xlaBackend.PluginName())
		return
	}

	plugins := []string{"cpu", "cuda", "tpu"}
	for _, plugin := range plugins {
		t.Run(plugin, func(t *testing.T) {
			backendName := fmt.Sprintf("%s:%s", xla.BackendName, plugin)
			if err := os.Setenv(compute.ConfigEnvVar, backendName); err != nil {
				t.Fatalf("Failed to set env %s=%s", compute.ConfigEnvVar, backendName)
			}
			defer os.Unsetenv(compute.ConfigEnvVar)

			backend, err := compute.New()
			if err != nil {
				t.Skipf("Plugin %q not available: %v", plugin, err)
				return
			}
			defer backend.Finalize()
			fn(t, backend, plugin)
		})
	}
}

func TestCompileAndRun(t *testing.T) {
	testAllPlugins(t, func(t *testing.T, backend compute.Backend, plugin string) {
		// Just return a constant.
		y0, err := testutil.Exec1(backend, nil, func(f compute.Function, params []compute.Value) (compute.Value, error) {
			return f.Constant([]float32{-7})
		})
		assert.NoError(t, err)
		assert.Equal(t, float32(-7), y0)
	})
}

// TestCompliance runs all compute.Backend compliance tests.
func TestCompliance(t *testing.T) {
	testAllPlugins(t, func(t *testing.T, backend compute.Backend, plugin string) {
		cfg := &backendtest.AllTestsConfiguration{}
		if plugin == "cuda" {
			// CUDA only support float32 and uint8 convolutions (!?)
			cfg.ConvGeneralDTypes = []dtypes.DType{dtypes.Float32}
		}
		backendtest.RunAll(t, backend, cfg)
	})
}
