// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package xla

import (
	"math"
	"testing"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/dtypes/bfloat16"
	"github.com/gomlx/compute/shapes"
)

// getFusionBackend creates a cuda backend and probes FusedScaledDotProductAttention on a tiny
// causal bf16 input. Skips if the cuda plugin is unavailable or if cuDNN fMHA is not supported.
func getFusionBackend(t *testing.T) *Backend {
	t.Helper()
	be, err := NewWithOptions("cuda", nil)
	if err != nil {
		t.Skipf("[cuda] plugin unavailable: %v", err)
	}
	t.Cleanup(func() { be.Finalize() })

	// Probe: tiny causal bf16 forward to confirm cuDNN fMHA actually works on this host.
	const B, S, H, D = 1, 4, 2, 8
	bld := be.Builder("fmha_probe").(*Builder)
	fn := bld.Main().(*Function)
	qkvShape := shapes.Make(dtypes.BFloat16, B, S, H, D)
	q, err := fn.Parameter("q", qkvShape, nil)
	if err != nil {
		t.Fatalf("probe: parameter q: %v", err)
	}
	k, err := fn.Parameter("k", qkvShape, nil)
	if err != nil {
		t.Fatalf("probe: parameter k: %v", err)
	}
	v, err := fn.Parameter("v", qkvShape, nil)
	if err != nil {
		t.Fatalf("probe: parameter v: %v", err)
	}
	out, _, err := fn.FusedScaledDotProductAttention(q, k, v,
		compute.AxesLayoutBSHD, &compute.ScaledDotProductAttentionConfig{Scale: 1.0/math.Sqrt(float64(D)), Causal: true})
	if compute.IsNotImplemented(err) {
		t.Skipf("[cuda] cuDNN fMHA not supported on this host: %v", err)
	}
	if err != nil {
		t.Fatalf("probe: FusedScaledDotProductAttention: %v", err)
	}
	if err = fn.Return([]compute.Value{out}, nil); err != nil {
		t.Fatalf("probe: Return: %v", err)
	}
	if _, err = bld.Compile(); err != nil {
		t.Skipf("[cuda] cuDNN fMHA compile failed (unsupported cuDNN): %v", err)
	}
	return be
}

// assertFiniteBSHD reads buf (bf16) and fails if any element is NaN or Inf.
func assertFiniteBSHD(t *testing.T, buf compute.Buffer) {
	t.Helper()
	sh, err := buf.Shape()
	if err != nil {
		t.Fatalf("assertFiniteBSHD Shape: %v", err)
	}
	flat := make([]bfloat16.BFloat16, sh.Size())
	if err = buf.ToFlatData(flat); err != nil {
		t.Fatalf("assertFiniteBSHD ToFlatData: %v", err)
	}
	bad := 0
	for _, x := range flat {
		f := x.Float32()
		if math.IsNaN(float64(f)) || math.IsInf(float64(f), 0) {
			bad++
		}
	}
	if bad > 0 {
		t.Errorf("assertFiniteBSHD: %d/%d non-finite values", bad, len(flat))
	} else {
		t.Logf("assertFiniteBSHD: all %d outputs finite", len(flat))
	}
}

// readFlatBF16 extracts the flat bf16 slice from buf, failing the test on error.
func readFlatBF16(t *testing.T, buf compute.Buffer) ([]bfloat16.BFloat16, shapes.Shape) {
	t.Helper()
	sh, err := buf.Shape()
	if err != nil {
		t.Fatalf("readFlatBF16 Shape: %v", err)
	}
	flat := make([]bfloat16.BFloat16, sh.Size())
	if err = buf.ToFlatData(flat); err != nil {
		t.Fatalf("readFlatBF16 ToFlatData: %v", err)
	}
	return flat, sh
}

// assertSeqLenMasksOutput verifies that seqlen masking actually changed the output. It compares
// the padding positions (seqIdx >= shortLen) of batch element 1 (masked to shortLen) against the
// same positions of batch element 0 (full seqLen). With q=k=v=ones and PADDING_CAUSAL, cuDNN
// zeros/masks the padding-query outputs for the short batch element, producing values that differ
// from the active (non-zero) outputs of the full batch element. If seqlens were ignored, all
// positions across both elements would be identical and the assertion would fail.
//
// Layout: flat [B, S, H, D] row-major. shortLen < seqLen required.
func assertSeqLenMasksOutput(t *testing.T, flat []bfloat16.BFloat16, b, seqLen, h, d, shortLen int) {
	t.Helper()
	if shortLen >= seqLen {
		t.Fatalf("assertSeqLenMasksOutput: shortLen %d must be < seqLen %d", shortLen, seqLen)
	}
	stride := func(batch, seq int) int { return (batch*seqLen+seq)*h*d }
	// Count positions in [batch=1, s=shortLen..seqLen-1] that differ from the corresponding
	// position in [batch=0] (which has full seqlen and should be active/non-zero).
	diffCount := 0
	totalPad := (seqLen - shortLen) * h * d
	for s := shortLen; s < seqLen; s++ {
		base0 := stride(0, s)
		base1 := stride(1, s)
		for i := 0; i < h*d; i++ {
			if flat[base0+i] != flat[base1+i] {
				diffCount++
			}
		}
	}
	if diffCount == 0 {
		t.Errorf("assertSeqLenMasksOutput: padding positions (s>=%d) in batch 1 are identical to batch 0 (%d elements) -- seqlen masking had no effect", shortLen, totalPad)
	} else {
		t.Logf("assertSeqLenMasksOutput: %d/%d padding-position elements differ (masking active)", diffCount, totalPad)
	}
}

// [cuda] PADDING_CAUSAL: per-batch lengths shorter than S must mask the padding rows. With
// q=k=v=ones, masking changes which keys contribute, so the masked output differs from the
// unmasked all-ones output for the shortened batch element. Runs under xla:cuda.
func TestFMHA_SeqLenPaddingCausal_cuda(t *testing.T) {
	be := getFusionBackend(t)
	const B, S, H, D = 2, 8, 12, 64
	scale := 1.0 / math.Sqrt(float64(D))
	qkvShape := shapes.Make(dtypes.BFloat16, B, S, H, D)

	bld := be.Builder("fmha_seqlen").(*Builder)
	fn := bld.Main().(*Function)

	q, err := fn.Parameter("q", qkvShape, nil)
	if err != nil {
		t.Fatalf("parameter q: %v", err)
	}
	k, err := fn.Parameter("k", qkvShape, nil)
	if err != nil {
		t.Fatalf("parameter k: %v", err)
	}
	v, err := fn.Parameter("v", qkvShape, nil)
	if err != nil {
		t.Fatalf("parameter v: %v", err)
	}
	// Seqlen constants embedded in the graph: batch 0 uses full S, batch 1 is padded to S/2.
	qSeq, err := fn.Constant([]int32{S, S / 2}, B)
	if err != nil {
		t.Fatalf("constant qSeqLen: %v", err)
	}
	kvSeq, err := fn.Constant([]int32{S, S / 2}, B)
	if err != nil {
		t.Fatalf("constant kvSeqLen: %v", err)
	}
	cfg := &compute.ScaledDotProductAttentionConfig{
		QuerySeqLen:    qSeq,
		KeyValueSeqLen: kvSeq,
	}
	cfg.Scale = scale
	cfg.Causal = true
	out, _, err := fn.FusedScaledDotProductAttention(q, k, v,
		compute.AxesLayoutBSHD, cfg)
	if err != nil {
		t.Fatalf("FusedScaledDotProductAttention: %v", err)
	}
	if err = fn.Return([]compute.Value{out}, nil); err != nil {
		t.Fatalf("Return: %v", err)
	}
	exec, err := bld.Compile()
	if err != nil {
		t.Fatalf("Compile: %v", err)
	}

	nElems := B * S * H * D
	ones := make([]bfloat16.BFloat16, nElems)
	one := bfloat16.FromFloat32(1)
	for i := range ones {
		ones[i] = one
	}
	mkBuf := func() compute.Buffer {
		buf, e := be.BufferFromFlatData(0, ones, qkvShape)
		if e != nil {
			t.Fatalf("BufferFromFlatData: %v", e)
		}
		return buf
	}
	qb, kb, vb := mkBuf(), mkBuf(), mkBuf()
	defer func() {
		_ = qb.Finalize()
		_ = kb.Finalize()
		_ = vb.Finalize()
	}()

	outs, err := exec.Execute([]compute.Buffer{qb, kb, vb}, nil, 0)
	if err != nil {
		t.Fatalf("Execute: %v", err)
	}
	defer func() {
		for _, o := range outs {
			_ = o.Finalize()
		}
	}()

	// Original finiteness check: no NaN/Inf anywhere in the output.
	assertFiniteBSHD(t, outs[0])

	// Masking-effect check: seqlen masking must change the output for the short batch element.
	// Batch 0 has qSeqLen=S (all active); batch 1 has qSeqLen=S/2 (padding at s>=S/2).
	// cuDNN PADDING_CAUSAL zeros/masks query positions >= qSeqLen, so batch 1's padding
	// positions must differ from batch 0's (which are fully active and non-zero). If cuDNN
	// ignored the seqlens, both elements would produce identical all-ones outputs and this
	// assertion would fail.
	flat, _ := readFlatBF16(t, outs[0])
	assertSeqLenMasksOutput(t, flat, B, S, H, D, S/2)
}
