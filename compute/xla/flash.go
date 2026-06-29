// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package xla

import (
	"fmt"
	"strconv"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/shapes"
	"github.com/pkg/errors"
)

// cuDNN fused-attention (flash) StableHLO custom_call targets, backend_config, and layouts.
//
// These custom-calls, their backend_config, and the operand/result layouts below are not part of
// the public XLA op set. They are the same calls JAX emits for
// jax.nn.dot_product_attention(..., implementation="cudnn"). Provenance:
//   - target names + backend_config field mapping (CudnnfMHABackendConfig, _custom_name_maps):
//     https://github.com/jax-ml/jax/blob/main/jax/_src/cudnn/fused_attention_stablehlo.py
//   - the exact backend_config JSON and layouts here were captured by lowering that JAX call to
//     StableHLO (jax.jit(fn).lower(...).compiler_ir("stablehlo")) and reading the emitted
//     stablehlo.custom_call. The stats (log-sum-exp) output appears only in the jax.grad lowering.
//   - XLA custom_call contract (documents api_version=4; these calls use api_version=2):
//     https://openxla.org/xla/custom_call
//   - cuDNN fMHA performance context: https://github.com/jax-ml/jax/issues/24934
const (
	fmhaForwardTarget  = "__cudnn$fmhaSoftmax"
	fmhaBackwardTarget = "__cudnn$fmhaSoftmaxBackward"
)

// flashFwdBackendConfig builds the forward cudnn_fmha_backend_config for q,k,v [B,S,H,D]
// (score matrix [B,H,S,S]). Only the scale and score-matrix dims vary with shape.
func flashFwdBackendConfig(b, h, s int, scale float64) string {
	return flashBackendConfig(b, h, s, scale,
		`"bmm1_dot_dimension_numbers": {"lhs_contracting_dimensions": ["3"], "rhs_contracting_dimensions": ["3"], "lhs_batch_dimensions": ["0", "2"], "rhs_batch_dimensions": ["0", "2"]}, "bmm2_dot_dimension_numbers": {"lhs_contracting_dimensions": ["3"], "rhs_contracting_dimensions": ["1"], "lhs_batch_dimensions": ["0", "1"], "rhs_batch_dimensions": ["0", "2"]}`)
}

// flashBwdBackendConfig is the backward counterpart: the four backward-gemm dot_dimension_numbers.
func flashBwdBackendConfig(b, h, s int, scale float64) string {
	return flashBackendConfig(b, h, s, scale,
		`"bmm1_grad_gemm1_dot_dimension_numbers": {"lhs_contracting_dimensions": ["2"], "rhs_contracting_dimensions": ["1"], "lhs_batch_dimensions": ["0", "1"], "rhs_batch_dimensions": ["0", "2"]}, "bmm1_grad_gemm2_dot_dimension_numbers": {"lhs_contracting_dimensions": ["3"], "rhs_contracting_dimensions": ["1"], "lhs_batch_dimensions": ["0", "1"], "rhs_batch_dimensions": ["0", "2"]}, "bmm2_grad_gemm1_dot_dimension_numbers": {"lhs_contracting_dimensions": ["2"], "rhs_contracting_dimensions": ["1"], "lhs_batch_dimensions": ["0", "1"], "rhs_batch_dimensions": ["0", "2"]}, "bmm2_grad_gemm2_dot_dimension_numbers": {"lhs_contracting_dimensions": ["3"], "rhs_contracting_dimensions": ["3"], "lhs_batch_dimensions": ["0", "2"], "rhs_batch_dimensions": ["0", "2"]}`)
}

// flashBackendConfig builds a cudnn_fmha_backend_config. fmha_scale and the score-matrix
// dimensions [B,H,S,S] vary with shape; dotDimNumbers carries the bmm dot_dimension_numbers
// JSON, the only part that differs between the forward and backward custom-calls.
func flashBackendConfig(b, h, s int, scale float64, dotDimNumbers string) string {
	return fmt.Sprintf(`{"operation_queue_id": "0", "cudnn_fmha_backend_config": {"algorithm": {"algo_id": "0", "math_type": "TENSOR_OP_MATH", "tuning_knobs": {"17": "1", "24": "0"}, "is_cudnn_frontend": true, "workspace_size": "0"}, "fmha_scale": %s, "intermediate_tensor_shape": {"element_type": "BF16", "dimensions": ["%d", "%d", "%d", "%d"], "tuple_shapes": [], "layout": {"dim_level_types": [], "dim_unique": [], "dim_ordered": [], "minor_to_major": ["3", "2", "1", "0"], "tiles": [], "element_size_in_bits": "0", "memory_space": "0", "index_primitive_type": "PRIMITIVE_TYPE_INVALID", "pointer_primitive_type": "PRIMITIVE_TYPE_INVALID", "dynamic_shape_metadata_prefix_bytes": "0"}, "is_dynamic_dimension": [false, false, false, false]}, "is_flash_attention": true, "mask_type": "CAUSAL", %s, "dropout_rate": 0.0, "seed": 42, "sliding_window_length": 0, "max_seg_per_batch": 1, "is_paged_attention": false}}`,
		formatScale(scale), b, h, s, s, dotDimNumbers)
}

// formatScale renders a float as a JSON number (no quotes, shortest round-trip form).
func formatScale(scale float64) string {
	return strconv.FormatFloat(scale, 'g', -1, 64)
}

// flashSupported reports whether the cuDNN flash path can serve this call. The cuDNN fMHA
// custom-calls here are causal, bf16, BSHD-layout, equal-head only, on a cuda plugin; anything
// else returns ErrNotImplemented so the graph layer differentiates the decomposed attention.
func (f *Function) flashSupported(op string, mask compute.Value, numHeads, numKVHeads int, axesLayout compute.AxesLayout, causal bool) error {
	if !f.builder.backend.plugin.IsCUDA() {
		return errors.Wrapf(compute.ErrNotImplemented, "%s: cuDNN flash needs the cuda plugin, have %q", op, f.builder.backend.pluginName)
	}
	if !causal || mask != nil || axesLayout != compute.AxesLayoutBSHD || numKVHeads != numHeads {
		return errors.Wrapf(compute.ErrNotImplemented,
			"%s: cuDNN flash path supports causal, no explicit mask, BSHD layout, equal q/kv heads only (got causal=%v mask!=nil=%v layout=%v heads=%d/%d)",
			op, causal, mask != nil, axesLayout, numHeads, numKVHeads)
	}
	return nil
}

// bshdDims returns the [B,S,H,D] dimensions of a rank-4 BSHD value.
func (f *Function) bshdDims(op string, v compute.Value) (b, s, h, d int, err error) {
	nodes, err := f.verifyAndCastValues(op, v)
	if err != nil {
		return 0, 0, 0, 0, err
	}
	dims := nodes[0].shape.Dimensions
	if len(dims) != 4 {
		return 0, 0, 0, 0, errors.Errorf("%s: expected rank-4 BSHD tensor, got shape %v", op, dims)
	}
	return dims[0], dims[1], dims[2], dims[3], nil
}

// bf16 casts a value to bfloat16 (the cuDNN kernel's precision). No-op if already bf16.
func (f *Function) bf16(v compute.Value) (compute.Value, error) {
	return f.ConvertDType(v, dtypes.BFloat16)
}

// FusedScaledDotProductAttention runs the cuDNN flash forward. query/key/value are [B,S,H,D]
// (BSHD), bf16. It returns the [B,S,H,D] bf16 output and the [B,H,S] f32 softmax statistics
// (log-sum-exp) the flash backward needs. The [B,H,S,S] scores never materialize. On non-cuda
// plugins or unsupported option combinations it returns ErrNotImplemented.
func (f *Function) FusedScaledDotProductAttention(query, key, value, mask compute.Value, numHeads, numKVHeads int, axesLayout compute.AxesLayout, scale float64, causal bool, options *compute.ScaledDotProductAttentionConfig) (output, softmaxStats compute.Value, err error) {
	const op = "FusedScaledDotProductAttention"
	if err = f.flashSupported(op, mask, numHeads, numKVHeads, axesLayout, causal); err != nil {
		return nil, nil, err
	}
	b, s, h, d, err := f.bshdDims(op, query)
	if err != nil {
		return nil, nil, err
	}
	q, err := f.bf16(query)
	if err != nil {
		return nil, nil, err
	}
	k, err := f.bf16(key)
	if err != nil {
		return nil, nil, err
	}
	v, err := f.bf16(value)
	if err != nil {
		return nil, nil, err
	}
	bhsd := shapes.Make(dtypes.BFloat16, b, h, s, d)
	stats := shapes.Make(dtypes.Float32, b, h, s)
	scratch := shapes.Make(dtypes.Uint8, 0)
	// Forward operand layouts: q,k,v BSHD [3,2,1,0]. Result layouts: output BHSD [3,1,2,0],
	// stats [2,1,0], scratch u8 [0].
	fwdOperandLayouts := [][]int{{3, 2, 1, 0}, {3, 2, 1, 0}, {3, 2, 1, 0}}
	fwdResultLayouts := [][]int{{3, 1, 2, 0}, {2, 1, 0}, {0}}
	outs, err := f.customCall(fmhaForwardTarget, flashFwdBackendConfig(b, h, s, scale),
		fwdOperandLayouts, []shapes.Shape{bhsd, stats, scratch}, fwdResultLayouts, q, k, v)
	if err != nil {
		return nil, nil, err
	}
	// outs[0] is BHSD; transpose to BSHD to match the query layout. outs[1] is the f32 stats.
	output, err = f.Transpose(outs[0], 0, 2, 1, 3)
	if err != nil {
		return nil, nil, err
	}
	return output, outs[1], nil
}

// FusedScaledDotProductAttentionVJP runs the cuDNN flash backward. It threads the softmaxStats
// from the forward (plus the forward output and the output gradient dOutput) into the cuDNN
// backward custom-call, so the [B,H,S,S] scores never materialize in the backward either.
// Returns dQuery, dKey, dValue as [B,S,H,D] bf16.
func (f *Function) FusedScaledDotProductAttentionVJP(query, key, value, mask compute.Value, numHeads, numKVHeads int, axesLayout compute.AxesLayout, scale float64, causal bool, options *compute.ScaledDotProductAttentionConfig, output, softmaxStats, dOutput compute.Value) (dQuery, dKey, dValue compute.Value, err error) {
	const op = "FusedScaledDotProductAttentionVJP"
	if err = f.flashSupported(op, mask, numHeads, numKVHeads, axesLayout, causal); err != nil {
		return nil, nil, nil, err
	}
	b, s, h, d, err := f.bshdDims(op, query)
	if err != nil {
		return nil, nil, nil, err
	}
	q, err := f.bf16(query)
	if err != nil {
		return nil, nil, nil, err
	}
	k, err := f.bf16(key)
	if err != nil {
		return nil, nil, nil, err
	}
	v, err := f.bf16(value)
	if err != nil {
		return nil, nil, nil, err
	}
	out, err := f.bf16(output)
	if err != nil {
		return nil, nil, nil, err
	}
	dOut, err := f.bf16(dOutput)
	if err != nil {
		return nil, nil, nil, err
	}
	bhsd := shapes.Make(dtypes.BFloat16, b, h, s, d)
	scratch := shapes.Make(dtypes.Uint8, 0)
	// Backward operands: q,k,v BSHD, stats [2,1,0], dOutput BSHD, output BSHD.
	bwdOperandLayouts := [][]int{{3, 2, 1, 0}, {3, 2, 1, 0}, {3, 2, 1, 0}, {2, 1, 0}, {3, 2, 1, 0}, {3, 2, 1, 0}}
	// Backward results: dQ, dK, dV BHSD, scratch u8.
	bwdResultLayouts := [][]int{{3, 1, 2, 0}, {3, 1, 2, 0}, {3, 1, 2, 0}, {0}}
	grads, err := f.customCall(fmhaBackwardTarget, flashBwdBackendConfig(b, h, s, scale),
		bwdOperandLayouts, []shapes.Shape{bhsd, bhsd, bhsd, scratch}, bwdResultLayouts,
		q, k, v, softmaxStats, dOut, out)
	if err != nil {
		return nil, nil, nil, err
	}
	// grads[0..2] = dQ, dK, dV (BHSD); transpose to BSHD to match q,k,v.
	if dQuery, err = f.Transpose(grads[0], 0, 2, 1, 3); err != nil {
		return nil, nil, nil, err
	}
	if dKey, err = f.Transpose(grads[1], 0, 2, 1, 3); err != nil {
		return nil, nil, nil, err
	}
	if dValue, err = f.Transpose(grads[2], 0, 2, 1, 3); err != nil {
		return nil, nil, nil, err
	}
	return dQuery, dKey, dValue, nil
}
