// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package xla_test

import (
	"fmt"
	"os"
	"testing"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/support/backendtest"
	"github.com/gomlx/compute/support/testutil"
	"github.com/gomlx/go-xla/compute/xla"
	"github.com/stretchr/testify/assert"
	"k8s.io/klog/v2"
)

func init() {
	klog.InitFlags(nil)
}

func testAllPlugins(t *testing.T, fn func(t *testing.T, backend compute.Backend)) {
	envBackend := os.Getenv(compute.ConfigEnvVar)
	if envBackend != "" {
		backend, err := compute.New()
		if err != nil {
			t.Fatalf("Failed to create backend %q: %v", envBackend, err)
		}
		defer backend.Finalize()
		fn(t, backend)
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
			fn(t, backend)
		})
	}
}

func TestCompileAndRun(t *testing.T) {
	testAllPlugins(t, func(t *testing.T, backend compute.Backend) {
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
	testAllPlugins(t, func(t *testing.T, backend compute.Backend) {
		backendtest.RunAll(t, backend)
	})
}
