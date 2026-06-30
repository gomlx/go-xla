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

// cuDNN fused-attention custom-call targets. S1 wires the standard softmax pair only.
// [S2] (Task 2b) adds the fmhaScaleBias* / *Dropout / fmhaScaleBias*Dropout target rows.
// FP8 targets (__cudnn$fmhaSoftmaxF8 / …BackwardF8) are intentionally NOT defined:
// fp8 fmha is paused (no local sm_8.9+ hardware to test). fp8 input dtype falls to
// ErrNotImplemented in selectFMHAVariant. Add that row when wiring fp8 on Hopper/Ada.
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
	fmhaSoftmaxFwd = "__cudnn$fmhaSoftmax"
	fmhaSoftmaxBwd = "__cudnn$fmhaSoftmaxBackward"
)

// fmhaVariant captures the config-derived custom-call selection: the fwd/bwd targets, the
// backend_config mask_type, and the operand-set flags. Built by selectFMHAVariant.
// S1 fields only; [S2] (Task 2b) adds `dropoutRate float64` and `hasBias bool`.
type fmhaVariant struct {
	fwdTarget, bwdTarget string
	maskType             string // "CAUSAL" | "PADDING" | "PADDING_CAUSAL" | "NO_MASK"
	hasSeqLens           bool
}

// selectFMHAVariant maps the q/k/v dtype and (causal, seqlens) to a cuDNN variant. S1 routes the
// standard softmax target only. Dtype gate: f16/bf16 only; anything else (incl. fp8 e4m3fn/e5m2 --
// paused, no local hardware) -> ErrNotImplemented, and the caller falls back to the decomposed path.
// mask_type derives from causal + seqlens: PADDING_CAUSAL (both), PADDING (seqlens only),
// CAUSAL (causal only), NO_MASK (neither).
//
// S1 reads ONLY cfg.QuerySeqLen/cfg.KeyValueSeqLen (the compute Stage-1 fields). It must not touch
// cfg.Bias/cfg.DropoutRate/cfg.DropoutSeed/cfg.DropoutOffset -- those land in compute Stage 2 and
// are wired here by Task 2b [S2], which extends this function with the bias/dropout precedence.
func selectFMHAVariant(op string, qkvDType dtypes.DType, causal bool,
	cfg *compute.ScaledDotProductAttentionConfig) (fmhaVariant, error) {
	var v fmhaVariant
	hasSeqLens := cfg != nil && cfg.QuerySeqLen != nil && cfg.KeyValueSeqLen != nil

	switch qkvDType {
	case dtypes.Float16, dtypes.BFloat16:
		v.fwdTarget, v.bwdTarget = fmhaSoftmaxFwd, fmhaSoftmaxBwd
	default:
		// fp8 (e4m3fn/e5m2) lands here too: paused, not wired. NotImplemented -> decomposed.
		return v, errors.Wrapf(compute.ErrNotImplemented,
			"%s: cuDNN fmha needs f16/bf16, got %s", op, qkvDType)
	}

	switch {
	case causal && hasSeqLens:
		v.maskType = "PADDING_CAUSAL"
	case hasSeqLens:
		v.maskType = "PADDING"
	case causal:
		v.maskType = "CAUSAL"
	default:
		v.maskType = "NO_MASK"
	}
	v.hasSeqLens = hasSeqLens
	return v, nil
}

// flashBackendConfigV builds a cudnn_fmha_backend_config for the given variant: mask_type comes
// from v, the score-matrix dims [B,H,S,S] and fmha_scale from the shape, dotDimNumbers carries the
// bmm dot_dimension_numbers JSON (the fwd/bwd-specific part). S1 has no dropout, so dropout_rate is
// the literal 0; Task 2b [S2] switches it to formatScale(v.dropoutRate).
func flashBackendConfigV(b, h, s int, scale float64, dotDimNumbers string, v fmhaVariant) string {
	return fmt.Sprintf(`{"operation_queue_id": "0", "cudnn_fmha_backend_config": {"algorithm": {"algo_id": "0", "math_type": "TENSOR_OP_MATH", "tuning_knobs": {"17": "1", "24": "0"}, "is_cudnn_frontend": true, "workspace_size": "0"}, "fmha_scale": %s, "intermediate_tensor_shape": {"element_type": "BF16", "dimensions": ["%d", "%d", "%d", "%d"], "tuple_shapes": [], "layout": {"dim_level_types": [], "dim_unique": [], "dim_ordered": [], "minor_to_major": ["3", "2", "1", "0"], "tiles": [], "element_size_in_bits": "0", "memory_space": "0", "index_primitive_type": "PRIMITIVE_TYPE_INVALID", "pointer_primitive_type": "PRIMITIVE_TYPE_INVALID", "dynamic_shape_metadata_prefix_bytes": "0"}, "is_dynamic_dimension": [false, false, false, false]}, "is_flash_attention": true, "mask_type": "%s", %s, "dropout_rate": 0, "seed": 42, "sliding_window_length": 0, "max_seg_per_batch": 1, "is_paged_attention": false}}`,
		formatScale(scale), b, h, s, s, v.maskType, dotDimNumbers)
}

// flashFwdBackendConfig builds the forward cudnn_fmha_backend_config for q,k,v [B,S,H,D]
// (score matrix [B,H,S,S]). Only the scale and score-matrix dims vary with shape.
func flashFwdBackendConfig(b, h, s int, scale float64, v fmhaVariant) string {
	return flashBackendConfigV(b, h, s, scale,
		`"bmm1_dot_dimension_numbers": {"lhs_contracting_dimensions": ["3"], "rhs_contracting_dimensions": ["3"], "lhs_batch_dimensions": ["0", "2"], "rhs_batch_dimensions": ["0", "2"]}, "bmm2_dot_dimension_numbers": {"lhs_contracting_dimensions": ["3"], "rhs_contracting_dimensions": ["1"], "lhs_batch_dimensions": ["0", "1"], "rhs_batch_dimensions": ["0", "2"]}`, v)
}

// flashBwdBackendConfig is the backward counterpart: the four backward-gemm dot_dimension_numbers.
func flashBwdBackendConfig(b, h, s int, scale float64, v fmhaVariant) string {
	return flashBackendConfigV(b, h, s, scale,
		`"bmm1_grad_gemm1_dot_dimension_numbers": {"lhs_contracting_dimensions": ["2"], "rhs_contracting_dimensions": ["1"], "lhs_batch_dimensions": ["0", "1"], "rhs_batch_dimensions": ["0", "2"]}, "bmm1_grad_gemm2_dot_dimension_numbers": {"lhs_contracting_dimensions": ["3"], "rhs_contracting_dimensions": ["1"], "lhs_batch_dimensions": ["0", "1"], "rhs_batch_dimensions": ["0", "2"]}, "bmm2_grad_gemm1_dot_dimension_numbers": {"lhs_contracting_dimensions": ["2"], "rhs_contracting_dimensions": ["1"], "lhs_batch_dimensions": ["0", "1"], "rhs_batch_dimensions": ["0", "2"]}, "bmm2_grad_gemm2_dot_dimension_numbers": {"lhs_contracting_dimensions": ["3"], "rhs_contracting_dimensions": ["3"], "lhs_batch_dimensions": ["0", "2"], "rhs_batch_dimensions": ["0", "2"]}`, v)
}

// formatScale renders a float as a JSON number (no quotes, shortest round-trip form).
func formatScale(scale float64) string {
	return strconv.FormatFloat(scale, 'g', -1, 64)
}

// flashSupported reports whether the cuDNN flash path can serve this call. cuDNN fMHA here is
// f16/bf16 (fp8 paused), BSHD-layout, equal-head, on a cuda plugin. Causality and
// per-batch seqlen padding are supported (mask_type derives from them in selectFMHAVariant);
// an explicit materialized mask is not (use seqlens instead). Anything else -> ErrNotImplemented.
func (f *Function) flashSupported(op string, mask compute.Value, numHeads, numKVHeads int, axesLayout compute.AxesLayout, causal bool, options *compute.ScaledDotProductAttentionConfig) error {
	if !f.builder.backend.plugin.IsCUDA() {
		return errors.Wrapf(compute.ErrNotImplemented, "%s: cuDNN flash needs the cuda plugin, have %q", op, f.builder.backend.pluginName)
	}
	if mask != nil {
		return errors.Wrapf(compute.ErrNotImplemented,
			"%s: cuDNN flash path takes seqlens, not a materialized mask", op)
	}
	if axesLayout != compute.AxesLayoutBSHD || numKVHeads != numHeads {
		return errors.Wrapf(compute.ErrNotImplemented,
			"%s: cuDNN flash path supports BSHD layout, equal q/kv heads only (got layout=%v heads=%d/%d)",
			op, axesLayout, numHeads, numKVHeads)
	}
	// One of QuerySeqLen/KeyValueSeqLen set without the other is ambiguous.
	if options != nil && (options.QuerySeqLen != nil) != (options.KeyValueSeqLen != nil) {
		return errors.Wrapf(compute.ErrNotImplemented,
			"%s: cuDNN flash padding mask needs both QuerySeqLen and KeyValueSeqLen", op)
	}
	return nil
}

// dtypeOf returns the element dtype of a value (used to pick the cuDNN fmha variant before
// any bf16 cast).
func (f *Function) dtypeOf(op string, v compute.Value) (dtypes.DType, error) {
	nodes, err := f.verifyAndCastValues(op, v)
	if err != nil {
		return 0, err
	}
	return nodes[0].shape.DType, nil
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

// validateSeqLen checks that v is a rank-1 int32 vector of length == batch.
// name is used in the error message (e.g. "QuerySeqLen"). Returns a wrapped
// error on mismatch so callers can distinguish validation failures.
func validateSeqLen(name string, v compute.Value, batch int) error {
	n, ok := v.(*Node)
	if !ok {
		return errors.Errorf("seqlen %s: expected *Node, got %T", name, v)
	}
	sh := n.shape
	if sh.DType != dtypes.Int32 {
		return errors.Errorf("seqlen %s: must be int32, got %s", name, sh.DType)
	}
	if sh.Rank() != 1 {
		return errors.Errorf("seqlen %s: must be rank-1 [B], got rank %d (shape %v)", name, sh.Rank(), sh.Dimensions)
	}
	if sh.Dimensions[0] != batch {
		return errors.Errorf("seqlen %s: length %d != batch size %d", name, sh.Dimensions[0], batch)
	}
	return nil
}

// fwdResultLayouts: output BHSD [3,1,2,0], stats [2,1,0], scratch u8 [0].
var fwdResultLayouts = [][]int{{3, 1, 2, 0}, {2, 1, 0}, {0}}

// bwdResultLayouts: dQ, dK, dV BHSD, scratch u8.
var bwdResultLayouts = [][]int{{3, 1, 2, 0}, {3, 1, 2, 0}, {3, 1, 2, 0}, {0}}

// FusedScaledDotProductAttention runs the cuDNN flash forward. query/key/value are [B,S,H,D]
// (BSHD), bf16. It returns the [B,S,H,D] bf16 output and the [B,H,S] f32 softmax statistics
// (log-sum-exp) the flash backward needs. The [B,H,S,S] scores never materialize. On non-cuda
// plugins or unsupported option combinations it returns ErrNotImplemented.
func (f *Function) FusedScaledDotProductAttention(query, key, value, mask compute.Value, numHeads, numKVHeads int, axesLayout compute.AxesLayout, scale float64, causal bool, options *compute.ScaledDotProductAttentionConfig) (output, softmaxStats compute.Value, err error) {
	const op = "FusedScaledDotProductAttention"
	if err = f.flashSupported(op, mask, numHeads, numKVHeads, axesLayout, causal, options); err != nil {
		return nil, nil, err
	}
	qDType, err := f.dtypeOf(op, query)
	if err != nil {
		return nil, nil, err
	}
	variant, err := selectFMHAVariant(op, qDType, causal, options)
	if err != nil {
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
	// Operand order cuDNN expects: q, k, v, [seqQ, seqKV]. [S2] inserts [bias] before seqlens
	// and appends [dropout seed, offset] after.
	operands := []compute.Value{q, k, v}
	operandLayouts := [][]int{{3, 2, 1, 0}, {3, 2, 1, 0}, {3, 2, 1, 0}}
	if variant.hasSeqLens {
		if err = validateSeqLen("QuerySeqLen", options.QuerySeqLen, b); err != nil {
			return nil, nil, err
		}
		if err = validateSeqLen("KeyValueSeqLen", options.KeyValueSeqLen, b); err != nil {
			return nil, nil, err
		}
		operands = append(operands, options.QuerySeqLen, options.KeyValueSeqLen)
		operandLayouts = append(operandLayouts, nil, nil) // int32 [B], row-major
	}
	bhsd := shapes.Make(dtypes.BFloat16, b, h, s, d)
	stats := shapes.Make(dtypes.Float32, b, h, s)
	scratch := shapes.Make(dtypes.Uint8, 0)
	outs, err := f.customCall(variant.fwdTarget, flashFwdBackendConfig(b, h, s, scale, variant),
		operandLayouts, []shapes.Shape{bhsd, stats, scratch}, fwdResultLayouts, operands...)
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
	if err = f.flashSupported(op, mask, numHeads, numKVHeads, axesLayout, causal, options); err != nil {
		return nil, nil, nil, err
	}
	qDType, err := f.dtypeOf(op, query)
	if err != nil {
		return nil, nil, nil, err
	}
	variant, err := selectFMHAVariant(op, qDType, causal, options)
	if err != nil {
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
	operands := []compute.Value{q, k, v, softmaxStats, dOut, out}
	operandLayouts := [][]int{{3, 2, 1, 0}, {3, 2, 1, 0}, {3, 2, 1, 0}, {2, 1, 0}, {3, 2, 1, 0}, {3, 2, 1, 0}}
	if variant.hasSeqLens {
		if err = validateSeqLen("QuerySeqLen", options.QuerySeqLen, b); err != nil {
			return nil, nil, nil, err
		}
		if err = validateSeqLen("KeyValueSeqLen", options.KeyValueSeqLen, b); err != nil {
			return nil, nil, nil, err
		}
		operands = append(operands, options.QuerySeqLen, options.KeyValueSeqLen)
		operandLayouts = append(operandLayouts, nil, nil)
	}
	bhsd := shapes.Make(dtypes.BFloat16, b, h, s, d)
	scratch := shapes.Make(dtypes.Uint8, 0)
	grads, err := f.customCall(variant.bwdTarget, flashBwdBackendConfig(b, h, s, scale, variant),
		operandLayouts, []shapes.Shape{bhsd, bhsd, bhsd, scratch}, bwdResultLayouts, operands...)
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
