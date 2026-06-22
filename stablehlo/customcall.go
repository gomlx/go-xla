package stablehlo

import (
	"github.com/gomlx/go-xla/internal/optypes"
	"github.com/gomlx/go-xla/types/shapes"
	"github.com/pkg/errors"
)

// CustomCallAPIVersionStatusReturning is XLA custom-call API version 2: the
// custom-call returns its outputs and an XLA status. It is what the cuDNN fused
// attention custom-calls use.
const CustomCallAPIVersionStatusReturning = 2

// CustomCall emits a stablehlo.custom_call to the named target (e.g.
// "__cudnn$fmhaSoftmax").
//
//   - apiVersion: the XLA custom-call API version (2 = STATUS_RETURNING).
//   - backendConfig: the raw backend_config string (e.g. a serialized proto / JSON).
//     Passed verbatim as a string attribute; "" omits it.
//   - operandLayouts, resultLayouts: pre-rendered MLIR array attributes constraining
//     the layouts, e.g. "[dense<[3, 2, 1, 0]> : tensor<4xindex>, ...]". "" omits them.
//   - outputShapes: one shape per result (the op is multi-output, e.g. attention
//     output plus a scratch workspace buffer).
//
// Returns one output Value per outputShape, in order.
func CustomCall(target string, apiVersion int, backendConfig, operandLayouts, resultLayouts string,
	outputShapes []shapes.Shape, operands ...*Value) ([]*Value, error) {
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

	stmt := fn.addMultiOp(op, outputShapes, operands)
	attrs := map[string]any{
		"call_target_name": target,
		"api_version":      int32(apiVersion),
	}
	if backendConfig != "" {
		attrs["backend_config"] = backendConfig
	}
	if operandLayouts != "" {
		attrs["operand_layouts"] = literalStr(operandLayouts)
	}
	if resultLayouts != "" {
		attrs["result_layouts"] = literalStr(resultLayouts)
	}
	stmt.Attributes = attrs
	return stmt.Outputs, nil
}
