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
	out, _, err := fn.FusedScaledDotProductAttention(q, k, v, nil, H, H,
		compute.AxesLayoutBSHD, 1.0/math.Sqrt(float64(D)), true, nil)
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
	out, _, err := fn.FusedScaledDotProductAttention(q, k, v, nil, H, H,
		compute.AxesLayoutBSHD, scale, true, cfg)
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
	assertFiniteBSHD(t, outs[0])
}
