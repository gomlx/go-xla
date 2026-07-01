// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package xla

import (
	"strings"
	"testing"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/shapes"
	"github.com/stretchr/testify/require"
)

// sent is a non-nil compute.Value sentinel. selectFMHAVariant only checks != nil.
var sent compute.Value = struct{}{}

// TestSelectFMHAVariant exercises the full dispatch table: dtype gate, causal/seqlens
// mask_type routing, bias target selection, and unsupported combinations. CPU-runnable.
func TestSelectFMHAVariant(t *testing.T) {
	cfgBias := &compute.ScaledDotProductAttentionConfig{Bias: sent}
	cfgSeqlens := &compute.ScaledDotProductAttentionConfig{
		QuerySeqLen:    sent,
		KeyValueSeqLen: sent,
	}
	cfgBiasSeqlens := &compute.ScaledDotProductAttentionConfig{
		Bias:           sent,
		QuerySeqLen:    sent,
		KeyValueSeqLen: sent,
	}

	cases := []struct {
		name        string
		dtype       dtypes.DType
		causal      bool
		cfg         *compute.ScaledDotProductAttentionConfig
		wantErr     bool
		wantNotImpl bool // error must be ErrNotImplemented
		wantFwd     string
		wantBwd     string
		wantMask    string
		wantBias    bool
		wantSeqlens bool
	}{
		{
			name:     "standard causal bf16",
			dtype:    dtypes.BFloat16,
			causal:   true,
			cfg:      nil,
			wantFwd:  fmhaSoftmaxFwd,
			wantBwd:  fmhaSoftmaxBwd,
			wantMask: "CAUSAL",
		},
		{
			name:     "no-mask non-causal f16",
			dtype:    dtypes.Float16,
			causal:   false,
			cfg:      nil,
			wantFwd:  fmhaSoftmaxFwd,
			wantBwd:  fmhaSoftmaxBwd,
			wantMask: "NO_MASK",
		},
		{
			name:        "rejects float32",
			dtype:       dtypes.Float32,
			causal:      true,
			cfg:         nil,
			wantErr:     true,
			wantNotImpl: true,
		},
		{
			name:        "rejects fp8 e4m3fn",
			dtype:       dtypes.F8E4M3FN,
			causal:      true,
			cfg:         nil,
			wantErr:     true,
			wantNotImpl: true,
		},
		{
			name:        "rejects fp8 e5m2",
			dtype:       dtypes.F8E5M2,
			causal:      true,
			cfg:         nil,
			wantErr:     true,
			wantNotImpl: true,
		},
		{
			name:        "seqlens non-causal -> PADDING",
			dtype:       dtypes.BFloat16,
			causal:      false,
			cfg:         cfgSeqlens,
			wantMask:    "PADDING",
			wantSeqlens: true,
			wantFwd:     fmhaSoftmaxFwd,
			wantBwd:     fmhaSoftmaxBwd,
		},
		{
			name:        "seqlens causal -> PADDING_CAUSAL",
			dtype:       dtypes.BFloat16,
			causal:      true,
			cfg:         cfgSeqlens,
			wantMask:    "PADDING_CAUSAL",
			wantSeqlens: true,
			wantFwd:     fmhaSoftmaxFwd,
			wantBwd:     fmhaSoftmaxBwd,
		},
		{
			name:     "bias routes to ScaleBias non-causal",
			dtype:    dtypes.BFloat16,
			causal:   false,
			cfg:      cfgBias,
			wantFwd:  fmhaScaleBiasSoftmaxFwd,
			wantBwd:  fmhaScaleBiasSoftmaxBwd,
			wantMask: "NO_MASK",
			wantBias: true,
		},
		{
			name:     "bias+causal routes to ScaleBias CAUSAL",
			dtype:    dtypes.BFloat16,
			causal:   true,
			cfg:      cfgBias,
			wantFwd:  fmhaScaleBiasSoftmaxFwd,
			wantBwd:  fmhaScaleBiasSoftmaxBwd,
			wantMask: "CAUSAL",
			wantBias: true,
		},
		{
			name:        "bias+seqlens not implemented",
			dtype:       dtypes.BFloat16,
			causal:      false,
			cfg:         cfgBiasSeqlens,
			wantErr:     true,
			wantNotImpl: true,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			v, err := selectFMHAVariant("op", tc.dtype, tc.causal, tc.cfg)
			if tc.wantErr {
				require.Error(t, err)
				if tc.wantNotImpl {
					require.True(t, compute.IsNotImplemented(err), "want ErrNotImplemented, got %v", err)
				}
				return
			}
			require.NoError(t, err)
			if tc.wantFwd != "" {
				require.Equal(t, tc.wantFwd, v.fwdTarget, "fwdTarget")
			}
			if tc.wantBwd != "" {
				require.Equal(t, tc.wantBwd, v.bwdTarget, "bwdTarget")
			}
			if tc.wantMask != "" {
				require.Equal(t, tc.wantMask, v.maskType, "maskType")
			}
			require.Equal(t, tc.wantBias, v.hasBias, "hasBias")
			require.Equal(t, tc.wantSeqlens, v.hasSeqLens, "hasSeqLens")
		})
	}
}

// nodeWithShape builds a minimal *Node with the given shape. value/builder are nil because
// validateSeqLen and validateBias only read n.shape -- no backend call is made.
func nodeWithShape(sh shapes.Shape) *Node { return &Node{shape: sh} }

// TestValidateSeqLen covers the CPU-runnable validation logic: wrong dtype, wrong rank,
// wrong length, and the happy path. No cuda or backend required.
func TestValidateSeqLen(t *testing.T) {
	const batch = 4

	cases := []struct {
		name        string
		node        *Node
		wantErr     bool
		errContains string
	}{
		{
			name:        "wrong dtype (bf16)",
			node:        nodeWithShape(shapes.Make(dtypes.BFloat16, batch)),
			wantErr:     true,
			errContains: "must be int32",
		},
		{
			name:        "wrong rank (rank-2)",
			node:        nodeWithShape(shapes.Make(dtypes.Int32, batch, 1)),
			wantErr:     true,
			errContains: "rank-1",
		},
		{
			name:        "wrong length",
			node:        nodeWithShape(shapes.Make(dtypes.Int32, batch+1)),
			wantErr:     true,
			errContains: "!= batch size",
		},
		{
			name:        "valid int32 [B]",
			node:        nodeWithShape(shapes.Make(dtypes.Int32, batch)),
			wantErr:     false,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			err := validateSeqLen("QuerySeqLen", tc.node, batch)
			if tc.wantErr {
				require.Error(t, err)
				if tc.errContains != "" {
					require.Contains(t, err.Error(), tc.errContains)
				}
			} else {
				require.NoError(t, err)
			}
		})
	}

	t.Run("not a *Node", func(t *testing.T) {
		var v compute.Value = struct{}{}
		err := validateSeqLen("QuerySeqLen", v, batch)
		require.Error(t, err)
		require.Contains(t, err.Error(), "*Node")
	})
}

func TestFlashBackendConfigV_MaskTypeFromVariant(t *testing.T) {
	v := fmhaVariant{maskType: "NO_MASK"}
	cfg := flashBackendConfigV(2, 12, 2048, 0.125, `"x": 1`, v)
	if !strings.Contains(cfg, `"mask_type": "NO_MASK"`) {
		t.Errorf("backend_config missing NO_MASK mask_type:\n%s", cfg)
	}
	if !strings.Contains(cfg, `"dropout_rate": 0`) {
		t.Errorf("backend_config missing dropout_rate:\n%s", cfg)
	}
}

// TestValidateBias covers the CPU-runnable bias validation: wrong type, wrong rank, wrong shape, happy path.
func TestValidateBias(t *testing.T) {
	const b, h, s, skv = 2, 4, 8, 8

	cases := []struct {
		name        string
		shape       shapes.Shape
		wantErr     bool
		errContains string
	}{
		{
			name:        "wrong dtype (int32)",
			shape:       shapes.Make(dtypes.Int32, b, h, s, skv),
			wantErr:     true,
			errContains: "half-precision or float32",
		},
		{
			name:        "wrong rank (rank-2)",
			shape:       shapes.Make(dtypes.BFloat16, b, h),
			wantErr:     true,
			errContains: "rank-4",
		},
		{
			name:        "wrong shape",
			shape:       shapes.Make(dtypes.BFloat16, b, h, s+1, skv),
			wantErr:     true,
			errContains: "must equal",
		},
		{
			name:    "valid bf16 [B,H,S,Skv]",
			shape:   shapes.Make(dtypes.BFloat16, b, h, s, skv),
			wantErr: false,
		},
		{
			name:    "valid float16 [B,H,S,Skv]",
			shape:   shapes.Make(dtypes.Float16, b, h, s, skv),
			wantErr: false,
		},
		{
			name:    "valid float32 [B,H,S,Skv]",
			shape:   shapes.Make(dtypes.Float32, b, h, s, skv),
			wantErr: false,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			v := nodeWithShape(tc.shape)
			err := validateBias("Bias", v, b, h, s, skv)
			if tc.wantErr {
				require.Error(t, err)
				if tc.errContains != "" {
					require.Contains(t, err.Error(), tc.errContains)
				}
			} else {
				require.NoError(t, err)
			}
		})
	}

	t.Run("not a *Node", func(t *testing.T) {
		var v compute.Value = struct{}{}
		err := validateBias("Bias", v, b, h, s, skv)
		require.Error(t, err)
		require.Contains(t, err.Error(), "*Node")
	})
}
