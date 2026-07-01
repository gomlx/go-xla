package stablehlo

import (
	"strconv"
	"strings"

	"github.com/gomlx/go-xla/internal/optypes"
	"github.com/gomlx/go-xla/types/shapes"
	"github.com/pkg/errors"
)

// CustomCallAPIVersionStatusReturning is XLA custom-call API version 2: the
// custom-call returns its outputs and an XLA status. It is what the cuDNN fused
// attention custom-calls use.
const CustomCallAPIVersionStatusReturning = 2

// renderLayouts renders the MLIR operand_layouts/result_layouts array attribute from
// minor-to-major dim orders. layouts==nil returns "" (omit the attribute). Each entry is
// paired by index with ranks; a nil/empty entry defaults to row-major (decreasing order
// over its rank), e.g. rank 4 -> [3, 2, 1, 0]. Output form:
//
//	"[dense<[3, 2, 1, 0]> : tensor<4xindex>, dense<0> : tensor<1xindex>]"
func renderLayouts(layouts [][]int, ranks []int) string {
	if layouts == nil {
		return ""
	}
	var sb strings.Builder
	sb.WriteByte('[')
	for i, rank := range ranks {
		if i > 0 {
			sb.WriteString(", ")
		}
		order := layouts[i]
		if len(order) == 0 {
			order = make([]int, rank)
			for j := range order {
				order[j] = rank - 1 - j
			}
		}
		sb.WriteString("dense<")
		if len(order) == 1 {
			sb.WriteString(strconv.Itoa(order[0]))
		} else {
			sb.WriteByte('[')
			for j, d := range order {
				if j > 0 {
					sb.WriteString(", ")
				}
				sb.WriteString(strconv.Itoa(d))
			}
			sb.WriteByte(']')
		}
		sb.WriteString("> : tensor<")
		sb.WriteString(strconv.Itoa(rank))
		sb.WriteString("xindex>")
	}
	sb.WriteByte(']')
	return sb.String()
}

// CustomCallV2 emits a stablehlo.custom_call (API version 2 = STATUS_RETURNING) to the named
// target (e.g. "__cudnn$fmhaSoftmax").
//
//   - backendConfig: the raw backend_config string (serialized proto / JSON); "" omits it.
//   - operandLayouts/outputLayouts: minor-to-major dim orders, paired by index with
//     operands/outputShapes.
//     Nil/empty TOP-LEVEL slice: the layout attribute is OMITTED entirely from the op.
//     Nil/empty ENTRY within a non-nil slice: that operand/result defaults to row-major
//     (decreasing order over its rank, e.g. rank 4 -> [3, 2, 1, 0]).
//   - outputShapes: one shape per result (multi-output: e.g. attention output + scratch).
//
// Returns one output Value per outputShape, in order.
func CustomCallV2(
	target string,
	backendConfig string,
	operands []*Value,
	operandLayouts [][]int,
	outputShapes []shapes.Shape,
	outputLayouts [][]int,
) ([]*Value, error) {
	op := optypes.CustomCall
	if len(operands) == 0 {
		return nil, errors.Errorf("%s requires at least one operand", op)
	}
	if len(outputShapes) == 0 {
		return nil, errors.Errorf("%s requires at least one output shape", op)
	}
	fn, err := innerMostFunction(operands...)
	if err != nil {
		return nil, err
	}
	if fn.Returned {
		return nil, errors.Errorf("cannot add operation %s after returning, in function %q", op, fn.Name)
	}

	operandRanks := make([]int, len(operands))
	for i, v := range operands {
		operandRanks[i] = v.shape.Rank()
	}
	outputRanks := make([]int, len(outputShapes))
	for i, s := range outputShapes {
		outputRanks[i] = s.Rank()
	}

	stmt := fn.addMultiOp(op, outputShapes, operands)
	attrs := map[string]any{
		"call_target_name": target,
		"api_version":      int32(CustomCallAPIVersionStatusReturning),
	}
	if backendConfig != "" {
		attrs["backend_config"] = backendConfig
	}
	if s := renderLayouts(operandLayouts, operandRanks); s != "" {
		attrs["operand_layouts"] = literalStr(s)
	}
	if s := renderLayouts(outputLayouts, outputRanks); s != "" {
		attrs["result_layouts"] = literalStr(s)
	}
	stmt.Attributes = attrs
	return stmt.Outputs, nil
}
