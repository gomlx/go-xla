// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package xla_test

import (
	"testing"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/support/testutil"
	"github.com/stretchr/testify/require"
)

// TestBinaryOp covers the different types of automatic broadcasting for binary operations.
func TestBinaryOp(t *testing.T) {
	testAllPlugins(t, func(t *testing.T, backend compute.Backend, plugin string) {
		runTestCase := func(name string, lhs, rhs, want any) {
			t.Run(name, func(t *testing.T) {
				result, err := testutil.Exec1(backend, []any{lhs, rhs}, func(f compute.Function, params []compute.Value) (compute.Value, error) {
					return f.Add(params[0], params[1])
				})
				require.NoError(t, err)
				require.Equal(t, want, result)
			})
		}

		runTestCase("same shape", float32(-2.0), float32(3.0), float32(1.0))
		runTestCase("lhs scalar", int32(-2), []int32{1, 5}, []int32{-1, 3})
		runTestCase("rhs scalar", []complex64{1i, 5i}, complex64(-2), []complex64{-2 + 1i, -2 + 5i})
		runTestCase("broadcast lhs to rhs", []int8{-1}, []int8{1, 5}, []int8{0, 4})
		runTestCase("broadcast both sides",
			[][]float64{{-1}, {-2}},
			[][]float64{{1, 5}},
			[][]float64{{0, 4}, {-1, 3}})

		t.Run("booleans", func(t *testing.T) {
			result, err := testutil.Exec1(backend, []any{
				[][]bool{{false}, {true}},
				[][]bool{{false, true}},
			}, func(f compute.Function, params []compute.Value) (compute.Value, error) {
				return f.LogicalAnd(params[0], params[1])
			})
			require.NoError(t, err)
			require.Equal(t,
				[][]bool{{false, false}, {false, true}},
				result)

		})
	})
}
