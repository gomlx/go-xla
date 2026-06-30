// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package xla

import (
	"errors"
	"strings"
	"testing"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/shapes"
	"github.com/stretchr/testify/require"
)

func TestSelectFMHAVariant_StandardCausal(t *testing.T) {
	v, err := selectFMHAVariant("op", dtypes.BFloat16, true, nil)
	if err != nil {
		t.Fatalf("unexpected err: %v", err)
	}
	if v.fwdTarget != "__cudnn$fmhaSoftmax" || v.bwdTarget != "__cudnn$fmhaSoftmaxBackward" {
		t.Errorf("targets = %q / %q", v.fwdTarget, v.bwdTarget)
	}
	if v.maskType != "CAUSAL" {
		t.Errorf("maskType = %q, want CAUSAL", v.maskType)
	}
}

func TestSelectFMHAVariant_NoMaskWhenNotCausal(t *testing.T) {
	v, err := selectFMHAVariant("op", dtypes.Float16, false, nil)
	if err != nil {
		t.Fatalf("unexpected err: %v", err)
	}
	if v.maskType != "NO_MASK" {
		t.Errorf("maskType = %q, want NO_MASK", v.maskType)
	}
}

func TestSelectFMHAVariant_RejectsF32(t *testing.T) {
	_, err := selectFMHAVariant("op", dtypes.Float32, true, nil)
	if !errors.Is(err, compute.ErrNotImplemented) {
		t.Errorf("err = %v, want ErrNotImplemented", err)
	}
}

// TestSelectFMHAVariant_FP8NotImplemented pins the fp8-paused seam: fp8 dtypes must return
// ErrNotImplemented so the caller falls back to the decomposed path. CPU, Mac-runnable.
func TestSelectFMHAVariant_FP8NotImplemented(t *testing.T) {
	_, err := selectFMHAVariant("fmha", dtypes.F8E4M3FN, true, nil)
	require.True(t, compute.IsNotImplemented(err), "fp8 must be NotImplemented (paused), got %v", err)
	_, err = selectFMHAVariant("fmha", dtypes.F8E5M2, true, nil)
	require.True(t, compute.IsNotImplemented(err), "fp8 e5m2 must be NotImplemented (paused), got %v", err)
}

// TestSelectFMHAVariant_SeqLenPadding confirms that both-seqlens routes to PADDING (non-causal)
// or PADDING_CAUSAL (causal), and nil cfg still routes to CAUSAL. CPU-runnable.
func TestSelectFMHAVariant_SeqLenPadding(t *testing.T) {
	// sentinel is a non-nil compute.Value (any). selectFMHAVariant only checks != nil.
	var sent compute.Value = struct{}{}

	cfgBoth := &compute.ScaledDotProductAttentionConfig{
		QuerySeqLen:    sent,
		KeyValueSeqLen: sent,
	}

	v, err := selectFMHAVariant("op", dtypes.BFloat16, false, cfgBoth)
	if err != nil {
		t.Fatalf("unexpected err: %v", err)
	}
	if v.maskType != "PADDING" {
		t.Errorf("maskType = %q, want PADDING", v.maskType)
	}
	if !v.hasSeqLens {
		t.Errorf("hasSeqLens = false, want true")
	}

	v, err = selectFMHAVariant("op", dtypes.BFloat16, true, cfgBoth)
	if err != nil {
		t.Fatalf("unexpected err: %v", err)
	}
	if v.maskType != "PADDING_CAUSAL" {
		t.Errorf("maskType = %q, want PADDING_CAUSAL", v.maskType)
	}

	// nil cfg still routes causal -> CAUSAL (no regression).
	v, err = selectFMHAVariant("op", dtypes.BFloat16, true, nil)
	if err != nil {
		t.Fatalf("unexpected err: %v", err)
	}
	if v.maskType != "CAUSAL" {
		t.Errorf("maskType = %q, want CAUSAL", v.maskType)
	}
}

// nodeWithShape builds a minimal *Node with the given shape. value/builder are nil because
// validateSeqLen only reads n.shape -- no backend call is made.
func nodeWithShape(sh shapes.Shape) *Node { return &Node{shape: sh} }

// TestValidateSeqLen covers the CPU-runnable validation logic: wrong dtype, wrong rank,
// wrong length, and the happy path. No cuda or backend required.
func TestValidateSeqLen(t *testing.T) {
	const batch = 4

	t.Run("wrong dtype (bf16)", func(t *testing.T) {
		v := nodeWithShape(shapes.Make(dtypes.BFloat16, batch))
		err := validateSeqLen("QuerySeqLen", v, batch)
		require.Error(t, err)
		require.Contains(t, err.Error(), "must be int32")
	})

	t.Run("wrong rank (rank-2)", func(t *testing.T) {
		v := nodeWithShape(shapes.Make(dtypes.Int32, batch, 1))
		err := validateSeqLen("KeyValueSeqLen", v, batch)
		require.Error(t, err)
		require.Contains(t, err.Error(), "rank-1")
	})

	t.Run("wrong length", func(t *testing.T) {
		v := nodeWithShape(shapes.Make(dtypes.Int32, batch+1))
		err := validateSeqLen("QuerySeqLen", v, batch)
		require.Error(t, err)
		require.Contains(t, err.Error(), "!= batch size")
	})

	t.Run("not a *Node", func(t *testing.T) {
		var v compute.Value = struct{}{}
		err := validateSeqLen("QuerySeqLen", v, batch)
		require.Error(t, err)
		require.Contains(t, err.Error(), "*Node")
	})

	t.Run("valid int32 [B]", func(t *testing.T) {
		v := nodeWithShape(shapes.Make(dtypes.Int32, batch))
		require.NoError(t, validateSeqLen("QuerySeqLen", v, batch))
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

// TestSelectFMHAVariant_BiasRoutes checks that a non-nil cfg.Bias selects the ScaleBias targets
// and sets hasBias. CPU-runnable.
func TestSelectFMHAVariant_BiasRoutes(t *testing.T) {
	var sent compute.Value = struct{}{}
	cfg := &compute.ScaledDotProductAttentionConfig{Bias: sent}

	v, err := selectFMHAVariant("op", dtypes.BFloat16, false, cfg)
	require.NoError(t, err)
	if v.fwdTarget != fmhaScaleBiasSoftmaxFwd {
		t.Errorf("fwdTarget = %q, want %q", v.fwdTarget, fmhaScaleBiasSoftmaxFwd)
	}
	if v.bwdTarget != fmhaScaleBiasSoftmaxBwd {
		t.Errorf("bwdTarget = %q, want %q", v.bwdTarget, fmhaScaleBiasSoftmaxBwd)
	}
	if !v.hasBias {
		t.Errorf("hasBias = false, want true")
	}
	if v.maskType != "NO_MASK" {
		t.Errorf("maskType = %q, want NO_MASK", v.maskType)
	}
}

// TestSelectFMHAVariant_BiasAndSeqlensNotImplemented checks that bias+seqlens returns
// ErrNotImplemented (cuDNN ScaleBias kernel does not accept seqlen operands). CPU-runnable.
func TestSelectFMHAVariant_BiasAndSeqlensNotImplemented(t *testing.T) {
	var sent compute.Value = struct{}{}
	cfg := &compute.ScaledDotProductAttentionConfig{
		Bias:           sent,
		QuerySeqLen:    sent,
		KeyValueSeqLen: sent,
	}
	_, err := selectFMHAVariant("op", dtypes.BFloat16, false, cfg)
	require.True(t, compute.IsNotImplemented(err), "bias+seqlens must be NotImplemented, got %v", err)
}

// TestSelectFMHAVariant_BiasCausal checks that bias+causal routes to ScaleBias with CAUSAL mask_type.
func TestSelectFMHAVariant_BiasCausal(t *testing.T) {
	var sent compute.Value = struct{}{}
	cfg := &compute.ScaledDotProductAttentionConfig{Bias: sent}

	v, err := selectFMHAVariant("op", dtypes.BFloat16, true, cfg)
	require.NoError(t, err)
	if v.fwdTarget != fmhaScaleBiasSoftmaxFwd {
		t.Errorf("fwdTarget = %q, want %q", v.fwdTarget, fmhaScaleBiasSoftmaxFwd)
	}
	if !v.hasBias {
		t.Errorf("hasBias = false, want true")
	}
	if v.maskType != "CAUSAL" {
		t.Errorf("maskType = %q, want CAUSAL", v.maskType)
	}
}

// TestValidateBias covers the CPU-runnable bias validation: wrong type, wrong rank, wrong shape, happy path.
func TestValidateBias(t *testing.T) {
	const b, h, s, skv = 2, 4, 8, 8

	t.Run("not a *Node", func(t *testing.T) {
		var v compute.Value = struct{}{}
		err := validateBias("Bias", v, b, h, s, skv)
		require.Error(t, err)
		require.Contains(t, err.Error(), "*Node")
	})

	t.Run("wrong dtype (int32)", func(t *testing.T) {
		v := nodeWithShape(shapes.Make(dtypes.Int32, b, h, s, skv))
		err := validateBias("Bias", v, b, h, s, skv)
		require.Error(t, err)
		require.Contains(t, err.Error(), "half-precision or float32")
	})

	t.Run("wrong rank (rank-2)", func(t *testing.T) {
		v := nodeWithShape(shapes.Make(dtypes.BFloat16, b, h))
		err := validateBias("Bias", v, b, h, s, skv)
		require.Error(t, err)
		require.Contains(t, err.Error(), "rank-4")
	})

	t.Run("wrong shape", func(t *testing.T) {
		v := nodeWithShape(shapes.Make(dtypes.BFloat16, b, h, s+1, skv))
		err := validateBias("Bias", v, b, h, s, skv)
		require.Error(t, err)
		require.Contains(t, err.Error(), "must equal")
	})

	t.Run("valid bf16 [B,H,S,Skv]", func(t *testing.T) {
		v := nodeWithShape(shapes.Make(dtypes.BFloat16, b, h, s, skv))
		require.NoError(t, validateBias("Bias", v, b, h, s, skv))
	})

	t.Run("valid float32 [B,H,S,Skv]", func(t *testing.T) {
		v := nodeWithShape(shapes.Make(dtypes.Float32, b, h, s, skv))
		require.NoError(t, validateBias("Bias", v, b, h, s, skv))
	})
}
