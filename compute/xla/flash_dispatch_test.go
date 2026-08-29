// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package xla

import (
	"errors"
	"strings"
	"testing"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/shapes"
)

func TestSelectFMHAVariant_StandardCausal(t *testing.T) {
	v, err := selectFMHAVariant("op", dtypes.BFloat16, &compute.ScaledDotProductAttentionConfig{Causal: true})
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
	v, err := selectFMHAVariant("op", dtypes.Float16, nil)
	if err != nil {
		t.Fatalf("unexpected err: %v", err)
	}
	if v.maskType != "NO_MASK" {
		t.Errorf("maskType = %q, want NO_MASK", v.maskType)
	}
}

func TestSelectFMHAVariant_RejectsF32(t *testing.T) {
	_, err := selectFMHAVariant("op", dtypes.Float32, &compute.ScaledDotProductAttentionConfig{Causal: true})
	if !errors.Is(err, compute.ErrNotImplemented) {
		t.Errorf("err = %v, want ErrNotImplemented", err)
	}
}

// TestSelectFMHAVariant_FP8NotImplemented pins the fp8-paused seam: fp8 dtypes must return
// ErrNotImplemented so the caller falls back to the decomposed path. CPU, Mac-runnable.
func TestSelectFMHAVariant_FP8NotImplemented(t *testing.T) {
	_, err := selectFMHAVariant("fmha", dtypes.F8E4M3FN, &compute.ScaledDotProductAttentionConfig{Causal: true})
	if !compute.IsNotImplemented(err) {
		t.Errorf("fp8 must be NotImplemented (paused), got %v", err)
	}
	_, err = selectFMHAVariant("fmha", dtypes.F8E5M2, &compute.ScaledDotProductAttentionConfig{Causal: true})
	if !compute.IsNotImplemented(err) {
		t.Errorf("fp8 e5m2 must be NotImplemented (paused), got %v", err)
	}
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

	v, err := selectFMHAVariant("op", dtypes.BFloat16, cfgBoth)
	if err != nil {
		t.Fatalf("unexpected err: %v", err)
	}
	if v.maskType != "PADDING" {
		t.Errorf("maskType = %q, want PADDING", v.maskType)
	}
	if !v.hasSeqLens {
		t.Errorf("hasSeqLens = false, want true")
	}

	cfgBoth.Causal = true
	v, err = selectFMHAVariant("op", dtypes.BFloat16, cfgBoth)
	if err != nil {
		t.Fatalf("unexpected err: %v", err)
	}
	if v.maskType != "PADDING_CAUSAL" {
		t.Errorf("maskType = %q, want PADDING_CAUSAL", v.maskType)
	}

	// nil cfg still routes causal -> CAUSAL (no regression).
	v, err = selectFMHAVariant("op", dtypes.BFloat16, &compute.ScaledDotProductAttentionConfig{Causal: true})
	if err != nil {
		t.Fatalf("unexpected err: %v", err)
	}
	if v.maskType != "CAUSAL" {
		t.Errorf("maskType = %q, want CAUSAL", v.maskType)
	}
}

// TestSelectFMHAVariant_Bias confirms cfg.Bias non-nil selects the fmhaScaleBias targets and sets
// hasBias, while mask_type still derives from causal. CPU-runnable.
func TestSelectFMHAVariant_Bias(t *testing.T) {
	var sent compute.Value = struct{}{}
	cfg := &compute.ScaledDotProductAttentionConfig{Bias: sent}

	cfg.Causal = true
	v, err := selectFMHAVariant("op", dtypes.BFloat16, cfg)
	if err != nil {
		t.Fatalf("unexpected err: %v", err)
	}
	if v.fwdTarget != fmhaScaleBiasSoftmaxFwd || v.bwdTarget != fmhaScaleBiasSoftmaxBwd {
		t.Errorf("targets = %q/%q, want ScaleBias", v.fwdTarget, v.bwdTarget)
	}
	if !v.hasBias {
		t.Errorf("hasBias = false, want true")
	}
	if v.maskType != "CAUSAL" {
		t.Errorf("maskType = %q, want CAUSAL", v.maskType)
	}
}

// TestSelectFMHAVariant_BiasSeqLensNotImplemented confirms bias+seqlens returns ErrNotImplemented
// (the cuDNN ScaleBias kernel takes no seqlen operands), so the caller falls back to decomposed.
func TestSelectFMHAVariant_BiasSeqLensNotImplemented(t *testing.T) {
	var sent compute.Value = struct{}{}
	cfg := &compute.ScaledDotProductAttentionConfig{Bias: sent, QuerySeqLen: sent, KeyValueSeqLen: sent}

	_, err := selectFMHAVariant("op", dtypes.BFloat16, cfg)
	if !compute.IsNotImplemented(err) {
		t.Errorf("err = %v, want ErrNotImplemented", err)
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
		if err == nil {
			t.Fatalf("expected error, got nil")
		}
		if !strings.Contains(err.Error(), "must be int32") {
			t.Errorf("expected error containing 'must be int32', got %q", err.Error())
		}
	})

	t.Run("wrong rank (rank-2)", func(t *testing.T) {
		v := nodeWithShape(shapes.Make(dtypes.Int32, batch, 1))
		err := validateSeqLen("KeyValueSeqLen", v, batch)
		if err == nil {
			t.Fatalf("expected error, got nil")
		}
		if !strings.Contains(err.Error(), "rank-1") {
			t.Errorf("expected error containing 'rank-1', got %q", err.Error())
		}
	})

	t.Run("wrong length", func(t *testing.T) {
		v := nodeWithShape(shapes.Make(dtypes.Int32, batch+1))
		err := validateSeqLen("QuerySeqLen", v, batch)
		if err == nil {
			t.Fatalf("expected error, got nil")
		}
		if !strings.Contains(err.Error(), "!= batch size") {
			t.Errorf("expected error containing '!= batch size', got %q", err.Error())
		}
	})

	t.Run("not a *Node", func(t *testing.T) {
		var v compute.Value = struct{}{}
		err := validateSeqLen("QuerySeqLen", v, batch)
		if err == nil {
			t.Fatalf("expected error, got nil")
		}
		if !strings.Contains(err.Error(), "*Node") {
			t.Errorf("expected error containing '*Node', got %q", err.Error())
		}
	})

	t.Run("valid int32 [B]", func(t *testing.T) {
		v := nodeWithShape(shapes.Make(dtypes.Int32, batch))
		if err := validateSeqLen("QuerySeqLen", v, batch); err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
	})
}

func TestFlashBackendConfigV_MaskTypeFromVariant(t *testing.T) {
	v := fmhaVariant{maskType: "NO_MASK", elementType: "BF16"}
	cfg := flashBackendConfigV(2, 12, 2048, 0.125, map[string]any{"x": 1}, v)
	cfg = strings.ReplaceAll(cfg, " ", "") // Remove spaces, to normalize.
	if !strings.Contains(cfg, `"mask_type":"NO_MASK"`) {
		t.Errorf("backend_config missing NO_MASK mask_type:\n%s", cfg)
	}
	if !strings.Contains(cfg, `"dropout_rate":0`) {
		t.Errorf("backend_config missing dropout_rate:\n%s", cfg)
	}
	// The intermediate element_type must reflect the variant's dtype (drives f16 vs bf16 backward).
	if !strings.Contains(cfg, `"element_type":"BF16"`) {
		t.Errorf("backend_config intermediate element_type != BF16:\n%s", cfg)
	}
}

// TestFlashBackendConfigV_ElementTypeFromDtype confirms selectFMHAVariant sets elementType so the
// backend_config intermediate tensor matches the q/k/v dtype (a mismatch fails the f16 backward).
func TestFlashBackendConfigV_ElementTypeFromDtype(t *testing.T) {
	for _, tc := range []struct {
		dtype dtypes.DType
		want  string
	}{
		{dtypes.BFloat16, "BF16"},
		{dtypes.Float16, "F16"},
	} {
		v, err := selectFMHAVariant("op", tc.dtype, &compute.ScaledDotProductAttentionConfig{Causal: true})
		if err != nil {
			t.Fatalf("unexpected error for %s: %v", tc.dtype, err)
		}
		if v.elementType != tc.want {
			t.Errorf("elementType for %s: got %q, want %q", tc.dtype, v.elementType, tc.want)
		}
		cfg := strings.ReplaceAll(flashBackendConfigV(1, 1, 8, 1.0, map[string]any{"x": 1}, v), " ", "")
		wantCfg := `"element_type":"` + tc.want + `"`
		if !strings.Contains(cfg, wantCfg) {
			t.Errorf("config element_type for %s: missing %q in %s", tc.dtype, wantCfg, cfg)
		}
	}
}
