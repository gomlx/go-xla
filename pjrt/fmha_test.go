package pjrt

import (
	"math"
	"testing"

	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/dtypes/bfloat16"
	"github.com/gomlx/go-xla/stablehlo"
	"github.com/gomlx/go-xla/types/shapes"
)

// destroyAll frees PJRT buffers (the device memory pool is held per client, so
// leaking them across tests in one process can starve a later test).
func destroyAll(bufs ...*Buffer) {
	for _, b := range bufs {
		if b != nil {
			_ = b.Destroy()
		}
	}
}

// Full backend_config that JAX 0.10.2 emits for dot_product_attention(cudnn) on
// q,k,v = bf16[2,2048,12,64], causal. Captured from the StableHLO reference.
const fmhaFwdBackendConfig = `{"operation_queue_id": "0", "cudnn_fmha_backend_config": {"algorithm": {"algo_id": "0", "math_type": "TENSOR_OP_MATH", "tuning_knobs": {"17": "1", "24": "0"}, "is_cudnn_frontend": true, "workspace_size": "0"}, "fmha_scale": 0.125, "intermediate_tensor_shape": {"element_type": "BF16", "dimensions": ["2", "12", "2048", "2048"], "tuple_shapes": [], "layout": {"dim_level_types": [], "dim_unique": [], "dim_ordered": [], "minor_to_major": ["3", "2", "1", "0"], "tiles": [], "element_size_in_bits": "0", "memory_space": "0", "index_primitive_type": "PRIMITIVE_TYPE_INVALID", "pointer_primitive_type": "PRIMITIVE_TYPE_INVALID", "dynamic_shape_metadata_prefix_bytes": "0"}, "is_dynamic_dimension": [false, false, false, false]}, "is_flash_attention": true, "mask_type": "CAUSAL", "bmm1_dot_dimension_numbers": {"lhs_contracting_dimensions": ["3"], "rhs_contracting_dimensions": ["3"], "lhs_batch_dimensions": ["0", "2"], "rhs_batch_dimensions": ["0", "2"]}, "bmm2_dot_dimension_numbers": {"lhs_contracting_dimensions": ["3"], "rhs_contracting_dimensions": ["1"], "lhs_batch_dimensions": ["0", "1"], "rhs_batch_dimensions": ["0", "2"]}, "dropout_rate": 0.0, "seed": 42, "sliding_window_length": 0, "max_seg_per_batch": 1, "is_paged_attention": false}}`

// TestFMHAForwardExecute compiles and runs the cuDNN flash-attention forward
// custom_call emitted by stablehlo.CustomCall, end to end on the GPU. With
// q=k=v=ones every score is equal so the softmax is uniform and the output is all
// 1.0 (the weighted sum of all-ones values), independent of the causal mask. Needs
// the cuda plugin (cuDNN); skipped otherwise.
func TestFMHAForwardExecute(t *testing.T) {
	if *FlagPluginName != "cuda" {
		t.Skipf("fmha is cuDNN-only; run with -plugin cuda (have %q)", *FlagPluginName)
	}
	client := getPJRTClient(t)
	const B, S, H, D = 2, 2048, 12, 64
	qkv := shapes.Make(dtypes.BFloat16, B, S, H, D)      // [B,S,H,D]
	out := shapes.Make(dtypes.BFloat16, B, H, S, D)      // [B,H,S,D]
	scratch := shapes.Make(dtypes.Uint8, 0)
	const layIn = "[dense<[3, 2, 1, 0]> : tensor<4xindex>, dense<[3, 2, 1, 0]> : tensor<4xindex>, dense<[3, 2, 1, 0]> : tensor<4xindex>]"
	const layOut = "[dense<[3, 1, 2, 0]> : tensor<4xindex>, dense<0> : tensor<1xindex>]"

	b := stablehlo.New("fmha_fwd")
	fn := b.Main()
	q := must1(fn.NamedInput("q", qkv))
	k := must1(fn.NamedInput("k", qkv))
	v := must1(fn.NamedInput("v", qkv))
	res := must1(stablehlo.CustomCall("__cudnn$fmhaSoftmax", stablehlo.CustomCallAPIVersionStatusReturning,
		fmhaFwdBackendConfig, layIn, layOut, []shapes.Shape{out, scratch}, q, k, v))
	must(fn.Return(res[0]))

	exec := must1(client.Compile().WithStableHLO(must1(b.Build())).Done())
	defer exec.Destroy()

	ones := make([]bfloat16.BFloat16, B*S*H*D)
	one := bfloat16.FromFloat32(1)
	for i := range ones {
		ones[i] = one
	}
	mk := func() *Buffer {
		return must1(client.BufferFromHost().FromFlatDataWithDimensions(ones, []int{B, S, H, D}).Done())
	}
	qb, kb, vb := mk(), mk(), mk()
	defer destroyAll(qb, kb, vb)
	results, err := exec.Execute(qb, kb, vb).DonateNone().Done()
	requireNoError(t, err)
	defer destroyAll(results...)
	assertLen(t, results, 1)

	flat, dims, err := results[0].ToFlatDataAndDimensions()
	requireNoError(t, err)
	t.Logf("flash forward ran, output dims=%v", dims)
	bf := flat.([]bfloat16.BFloat16)
	bad, first := 0, float32(0)
	for _, x := range bf {
		f := math.Float32frombits(uint32(x) << 16) // bf16 is the high 16 bits of f32
		if math.Abs(float64(f)-1.0) > 0.05 {
			if bad == 0 {
				first = f
			}
			bad++
		}
	}
	if bad > 0 {
		t.Errorf("%d/%d outputs not ~1.0 (first bad=%v); expected all 1.0", bad, len(bf), first)
	} else {
		t.Logf("OK: all %d outputs ~1.0", len(bf))
	}
}

// Full backward backend_config JAX emits for the gradient of the same attention.
const fmhaBwdBackendConfig = `{"operation_queue_id": "0", "cudnn_fmha_backend_config": {"algorithm": {"algo_id": "0", "math_type": "TENSOR_OP_MATH", "tuning_knobs": {"17": "1", "24": "0"}, "is_cudnn_frontend": true, "workspace_size": "0"}, "fmha_scale": 0.125, "intermediate_tensor_shape": {"element_type": "BF16", "dimensions": ["2", "12", "2048", "2048"], "tuple_shapes": [], "layout": {"dim_level_types": [], "dim_unique": [], "dim_ordered": [], "minor_to_major": ["3", "2", "1", "0"], "tiles": [], "element_size_in_bits": "0", "memory_space": "0", "index_primitive_type": "PRIMITIVE_TYPE_INVALID", "pointer_primitive_type": "PRIMITIVE_TYPE_INVALID", "dynamic_shape_metadata_prefix_bytes": "0"}, "is_dynamic_dimension": [false, false, false, false]}, "is_flash_attention": true, "mask_type": "CAUSAL", "bmm1_grad_gemm1_dot_dimension_numbers": {"lhs_contracting_dimensions": ["2"], "rhs_contracting_dimensions": ["1"], "lhs_batch_dimensions": ["0", "1"], "rhs_batch_dimensions": ["0", "2"]}, "bmm1_grad_gemm2_dot_dimension_numbers": {"lhs_contracting_dimensions": ["3"], "rhs_contracting_dimensions": ["1"], "lhs_batch_dimensions": ["0", "1"], "rhs_batch_dimensions": ["0", "2"]}, "bmm2_grad_gemm1_dot_dimension_numbers": {"lhs_contracting_dimensions": ["2"], "rhs_contracting_dimensions": ["1"], "lhs_batch_dimensions": ["0", "1"], "rhs_batch_dimensions": ["0", "2"]}, "bmm2_grad_gemm2_dot_dimension_numbers": {"lhs_contracting_dimensions": ["3"], "rhs_contracting_dimensions": ["3"], "lhs_batch_dimensions": ["0", "2"], "rhs_batch_dimensions": ["0", "2"]}, "dropout_rate": 0.0, "seed": 42, "sliding_window_length": 0, "max_seg_per_batch": 1, "is_paged_attention": false}}`

// TestFMHABackwardExecute chains the training forward (which also returns the
// softmax stats f32[B,H,S]) into __cudnn$fmhaSoftmaxBackward, all in one compiled
// graph: forward -> transpose output [B,H,S,D]->[B,S,H,D] -> backward, and runs it
// on the GPU. Confirms the backward custom_call lowers and executes and produces
// finite dQ/dK/dV. Needs the cuda plugin (cuDNN); skipped otherwise.
func TestFMHABackwardExecute(t *testing.T) {
	if *FlagPluginName != "cuda" {
		t.Skipf("fmha is cuDNN-only; run with -plugin cuda (have %q)", *FlagPluginName)
	}
	client := getPJRTClient(t)
	const B, S, H, D = 2, 2048, 12, 64
	bshd := shapes.Make(dtypes.BFloat16, B, S, H, D)
	bhsd := shapes.Make(dtypes.BFloat16, B, H, S, D)
	stats := shapes.Make(dtypes.Float32, B, H, S)
	scratch := shapes.Make(dtypes.Uint8, 0)
	const lay3bshd = "[dense<[3, 2, 1, 0]> : tensor<4xindex>, dense<[3, 2, 1, 0]> : tensor<4xindex>, dense<[3, 2, 1, 0]> : tensor<4xindex>]"
	const fwdResLay = "[dense<[3, 1, 2, 0]> : tensor<4xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<0> : tensor<1xindex>]"
	const bwdOpLay = "[dense<[3, 2, 1, 0]> : tensor<4xindex>, dense<[3, 2, 1, 0]> : tensor<4xindex>, dense<[3, 2, 1, 0]> : tensor<4xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<[3, 2, 1, 0]> : tensor<4xindex>, dense<[3, 2, 1, 0]> : tensor<4xindex>]"
	const bwdResLay = "[dense<[3, 1, 2, 0]> : tensor<4xindex>, dense<[3, 1, 2, 0]> : tensor<4xindex>, dense<[3, 1, 2, 0]> : tensor<4xindex>, dense<0> : tensor<1xindex>]"

	b := stablehlo.New("fmha_bwd")
	fn := b.Main()
	q := must1(fn.NamedInput("q", bshd))
	k := must1(fn.NamedInput("k", bshd))
	v := must1(fn.NamedInput("v", bshd))
	dO := must1(fn.NamedInput("dO", bshd))

	fwd := must1(stablehlo.CustomCall("__cudnn$fmhaSoftmax", stablehlo.CustomCallAPIVersionStatusReturning,
		fmhaFwdBackendConfig, lay3bshd, fwdResLay, []shapes.Shape{bhsd, stats, scratch}, q, k, v))
	outBSHD := must1(stablehlo.Transpose(fwd[0], 0, 2, 1, 3)) // [B,H,S,D] -> [B,S,H,D]
	grads := must1(stablehlo.CustomCall("__cudnn$fmhaSoftmaxBackward", stablehlo.CustomCallAPIVersionStatusReturning,
		fmhaBwdBackendConfig, bwdOpLay, bwdResLay, []shapes.Shape{bhsd, bhsd, bhsd, scratch}, q, k, v, fwd[1], dO, outBSHD))
	must(fn.Return(grads[0], grads[1], grads[2]))

	exec := must1(client.Compile().WithStableHLO(must1(b.Build())).Done())
	defer exec.Destroy()

	ones := make([]bfloat16.BFloat16, B*S*H*D)
	one := bfloat16.FromFloat32(1)
	for i := range ones {
		ones[i] = one
	}
	mk := func() *Buffer {
		return must1(client.BufferFromHost().FromFlatDataWithDimensions(ones, []int{B, S, H, D}).Done())
	}
	qb, kb, vb, dOb := mk(), mk(), mk(), mk()
	defer destroyAll(qb, kb, vb, dOb)
	results, err := exec.Execute(qb, kb, vb, dOb).DonateNone().Done()
	requireNoError(t, err)
	defer destroyAll(results...)
	assertLen(t, results, 3)

	for gi, name := range []string{"dQ", "dK", "dV"} {
		flat, _, err := results[gi].ToFlatDataAndDimensions()
		requireNoError(t, err)
		bf := flat.([]bfloat16.BFloat16)
		nan := 0
		for _, x := range bf {
			f := math.Float32frombits(uint32(x) << 16)
			if math.IsNaN(float64(f)) || math.IsInf(float64(f), 0) {
				nan++
			}
		}
		if nan > 0 {
			t.Errorf("%s: %d/%d non-finite", name, nan, len(bf))
		} else {
			t.Logf("%s: %d finite gradients", name, len(bf))
		}
	}
}
