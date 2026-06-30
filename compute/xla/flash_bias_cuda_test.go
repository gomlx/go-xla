// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package xla

import (
	"math"
	"testing"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/dtypes/bfloat16"
	"github.com/gomlx/compute/shapes"
	"github.com/stretchr/testify/require"
)

// TestFMHA_Bias_cuda runs the fused ScaleBias path with a strong additive bias that strongly
// favors key position 0 for all queries. The test then verifies that the bias-fused output
// differs from the no-bias output: if the bias operand were silently ignored, both outputs
// would be identical and the assertion would fail.
//
// Runs under xla:cuda; skips automatically if the cuda plugin or cuDNN fMHA is unavailable.
func TestFMHA_Bias_cuda(t *testing.T) {
	be := getFusionBackend(t)
	const B, S, H, D = 1, 4, 2, 8
	scale := 1.0 / math.Sqrt(float64(D))
	qkvShape := shapes.Make(dtypes.BFloat16, B, S, H, D)
	biasShape := shapes.Make(dtypes.BFloat16, B, H, S, S) // [B,H,S,Skv]

	// --- build a graph that returns both biased and unbiased outputs ---
	bld := be.Builder("fmha_bias").(*Builder)
	fn := bld.Main().(*Function)

	q, err := fn.Parameter("q", qkvShape, nil)
	require.NoError(t, err, "parameter q")
	k, err := fn.Parameter("k", qkvShape, nil)
	require.NoError(t, err, "parameter k")
	v, err := fn.Parameter("v", qkvShape, nil)
	require.NoError(t, err, "parameter v")
	bias, err := fn.Parameter("bias", biasShape, nil)
	require.NoError(t, err, "parameter bias")

	// unbiased forward
	outNoBias, _, err := fn.FusedScaledDotProductAttention(q, k, v, nil, H, H,
		compute.AxesLayoutBSHD, scale, false, nil)
	if compute.IsNotImplemented(err) {
		t.Skipf("[cuda] cuDNN fMHA not supported on this host: %v", err)
	}
	require.NoError(t, err, "FusedScaledDotProductAttention no-bias")

	// biased forward
	cfg := &compute.ScaledDotProductAttentionConfig{Bias: bias}
	outBias, _, err := fn.FusedScaledDotProductAttention(q, k, v, nil, H, H,
		compute.AxesLayoutBSHD, scale, false, cfg)
	require.NoError(t, err, "FusedScaledDotProductAttention with bias")

	require.NoError(t, fn.Return([]compute.Value{outNoBias, outBias}, nil))
	exec, err := bld.Compile()
	if err != nil {
		t.Skipf("[cuda] compile failed (cuDNN ScaleBias may be unsupported): %v", err)
	}

	// --- inputs ---
	// q=k=ones: all dot-product scores equal, so without bias every key gets equal weight.
	// V is non-uniform: V[b, key_i, h, d] = float(key_i+1), so attending to different keys
	// yields different outputs. With the bias strongly favoring key 0, biased output ≈ V[key 0]
	// and unbiased output ≈ mean(V) -- the two differ, proving the bias is live.
	nElemsQKV := B * S * H * D
	ones := make([]bfloat16.BFloat16, nElemsQKV) // q and k
	one := bfloat16.FromFloat32(1)
	for i := range ones {
		ones[i] = one
	}

	// V[b, s, h, d] = float32(s+1): value grows with key position (BSHD layout).
	vData := make([]bfloat16.BFloat16, nElemsQKV)
	for b := 0; b < B; b++ {
		for s := 0; s < S; s++ {
			for h := 0; h < H; h++ {
				for d := 0; d < D; d++ {
					idx := ((b*S+s)*H+h)*D + d
					vData[idx] = bfloat16.FromFloat32(float32(s + 1))
				}
			}
		}
	}

	// bias[b, h, s, 0] = +10, everything else = -10 => softmax sharply picks key 0
	nElemsBias := B * H * S * S
	biasData := make([]bfloat16.BFloat16, nElemsBias)
	neg := bfloat16.FromFloat32(-10)
	pos := bfloat16.FromFloat32(10)
	for i := range biasData {
		biasData[i] = neg
	}
	// set column 0 of every [S, S] slice to +10
	for b := 0; b < B; b++ {
		for h := 0; h < H; h++ {
			for s := 0; s < S; s++ {
				idx := ((b*H + h) * S * S) + s*S + 0
				biasData[idx] = pos
			}
		}
	}

	mkQ := func() compute.Buffer {
		buf, e := be.BufferFromFlatData(0, ones, qkvShape)
		require.NoError(t, e, "BufferFromFlatData q")
		return buf
	}
	mkV := func() compute.Buffer {
		buf, e := be.BufferFromFlatData(0, vData, qkvShape)
		require.NoError(t, e, "BufferFromFlatData v")
		return buf
	}
	// Keep mkQKV for k (same as q).
	mkQKV := mkQ
	mkBias := func() compute.Buffer {
		buf, e := be.BufferFromFlatData(0, biasData, biasShape)
		require.NoError(t, e, "BufferFromFlatData bias")
		return buf
	}

	qb, kb, vb, bb := mkQKV(), mkQKV(), mkV(), mkBias()
	defer func() {
		_ = qb.Finalize()
		_ = kb.Finalize()
		_ = vb.Finalize()
		_ = bb.Finalize()
	}()

	outs, err := exec.Execute([]compute.Buffer{qb, kb, vb, bb}, nil, 0)
	require.NoError(t, err, "Execute")
	defer func() {
		for _, o := range outs {
			_ = o.Finalize()
		}
	}()

	// assert both outputs are finite
	assertFiniteBSHD(t, outs[0])
	assertFiniteBSHD(t, outs[1])

	// assert that the bias changed the output; if ignored, outputs would be identical
	flatNoBias, _ := readFlatBF16(t, outs[0])
	flatBias, _ := readFlatBF16(t, outs[1])

	diffCount := 0
	for i := range flatNoBias {
		if flatNoBias[i] != flatBias[i] {
			diffCount++
		}
	}
	require.Greater(t, diffCount, 0,
		"bias-fused output is identical to no-bias output (%d elements) -- bias operand was ignored",
		len(flatNoBias))
	t.Logf("bias changed %d/%d output elements (bias operand is live)", diffCount, len(flatNoBias))
}
