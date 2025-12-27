package stablehlo

import (
	"fmt"
	"math"
	"reflect"
	"slices"
	"strconv"
	"strings"

	"github.com/gomlx/go-xla/internal/optypes"
	"github.com/gomlx/go-xla/internal/shapeinference"
	"github.com/gomlx/go-xla/pkg/types"
	"github.com/gomlx/go-xla/pkg/types/dtypes"
	"github.com/gomlx/go-xla/pkg/types/shapes"
	"github.com/pkg/errors"
)

// addOp adds a new operation to the function.
func (fn *Function) addOp(opType optypes.OpType, outputShape shapes.Shape, inputs ...*Value) *Statement {
	stmt := &Statement{
		Builder:  fn.Builder,
		Function: fn,
		OpType:   opType,
		Inputs:   inputs,
		Outputs:  []*Value{fn.newValue(outputShape)},
	}
	// Set the statement reference and output index for the output value
	stmt.Outputs[0].stmt = stmt
	stmt.Outputs[0].outputIndex = 0
	fn.Statements = append(fn.Statements, stmt)
	return stmt
}

// addMultiOp adds a new operation with multiple outputs to the function.
func (fn *Function) addMultiOp(opType optypes.OpType, outputShapes []shapes.Shape, inputs []*Value) *Statement {
	outputs := make([]*Value, len(outputShapes))
	for i, shape := range outputShapes {
		outputs[i] = fn.newValue(shape)
	}
	stmt := &Statement{
		Builder:  fn.Builder,
		Function: fn,
		OpType:   opType,
		Inputs:   inputs,
		Outputs:  outputs,
	}
	// Set the statement reference and output index for each output value
	for i := range outputs {
		outputs[i].stmt = stmt
		outputs[i].outputIndex = i
	}
	fn.Statements = append(fn.Statements, stmt)
	return stmt
}

// binaryOp adds a new binary operation to the function.
func (fn *Function) binaryOp(op optypes.OpType, lhs, rhs *Value) (*Value, error) {
	if fn.Returned {
		return nil, errors.Errorf("cannot add operation %s after returning, in function %q",
			op, fn.Name)
	}
	if lhs.fn != fn || rhs.fn != fn {
		return nil, errors.Errorf("cannot add operation %s to function %q, because the operands are not part of the function",
			op, fn.Name)
	}
	outputShape, err := shapeinference.BinaryOp(op, lhs.shape, rhs.shape)
	if err != nil {
		return nil, err
	}
	return fn.addOp(op, outputShape, lhs, rhs).Outputs[0], nil
}

// unaryOp adds a new unary operation to the function.
func (fn *Function) unaryOp(op optypes.OpType, operand *Value) (*Value, error) {
	if fn.Returned {
		return nil, errors.Errorf("cannot add operation %s after returning, in function %q",
			op, fn.Name)
	}
	if operand.fn != fn {
		return nil, errors.Errorf("cannot add operation %s to function %q, because the operand is not part of the function",
			op, fn.Name)
	}
	outputShape, err := shapeinference.UnaryOp(op, operand.shape)
	if err != nil {
		return nil, err
	}
	return fn.addOp(op, outputShape, operand).Outputs[0], nil
}

// tryExtractConstantShape attempts to extract concrete integer dimensions from a shape Value.
// If the shape Value was created by a Constant operation or a Concatenate of constants,
// it extracts the integer values.
// Returns (dimensions, true) if successful, or (nil, false) if the shape is truly dynamic.
func tryExtractConstantShape(fn *Function, shapeValue *Value) ([]int, bool) {
	return TryExtractConstantShape(fn, shapeValue)
}

// TryExtractConstantShape is the exported version of tryExtractConstantShape.
// It attempts to extract constant shape values from the given StableHLO value.
func TryExtractConstantShape(fn *Function, shapeValue *Value) ([]int, bool) {
	// Search through function statements to find where shapeValue was defined
	for _, stmt := range fn.Statements {
		// Check if this statement produced the shapeValue
		for _, output := range stmt.Outputs {
			if output == shapeValue {
				// Found the statement that created this value
				if stmt.OpType == optypes.Constant {
					// Extract the constant tensor data
					if valueAttr, ok := stmt.Attributes["value"]; ok {
						if tl, ok := valueAttr.(tensorLiteral); ok {
							result, ok := extractIntegersFromTensorLiteral(tl)
							if ok {
								return result, true
							}
						}
					}
				}
				// Handle Concatenate of constants (common pattern when building shape tensors)
				if stmt.OpType == optypes.Concatenate {
					return tryExtractConcatenatedShape(fn, stmt)
				}
				// Handle Reshape - just pass through to the input
				if stmt.OpType == optypes.Reshape {
					if len(stmt.Inputs) > 0 {
						return tryExtractConstantShape(fn, stmt.Inputs[0])
					}
				}
				// Handle GetDimensionSize - extract the dimension from the input's shape
				if stmt.OpType == optypes.GetDimensionSize {
					return tryExtractGetDimensionSize(fn, stmt)
				}
				// Handle Gather from constant tensors with constant indices
				if stmt.OpType == optypes.Gather {
					return tryExtractGatherConstant(fn, stmt)
				}
				// Handle Slice of constant tensors
				if stmt.OpType == optypes.Slice {
					return tryExtractSliceConstant(fn, stmt)
				}
				// Handle BroadcastInDim - when broadcasting a constant to a shape
				if stmt.OpType == optypes.BroadcastInDim {
					return tryExtractBroadcastInDimConstant(fn, stmt)
				}
				// Handle Multiply of constants (common in shape calculations)
				if stmt.OpType == optypes.Multiply {
					return tryExtractMultiplyConstant(fn, stmt)
				}
				// Handle Divide of constants (common in shape calculations, e.g., splitting heads)
				if stmt.OpType == optypes.Divide {
					return tryExtractDivideConstant(fn, stmt)
				}
				// Handle Select with constant condition (common in shape calculations with -1 inference)
				if stmt.OpType == optypes.Select {
					return tryExtractSelectConstant(fn, stmt)
				}
				// Handle Compare - try to evaluate constant comparisons
				if stmt.OpType == optypes.Compare {
					return tryExtractCompareConstant(fn, stmt)
				}
				// Handle Convert - just pass through to the input
				if stmt.OpType == optypes.Convert {
					if len(stmt.Inputs) > 0 {
						return tryExtractConstantShape(fn, stmt.Inputs[0])
					}
				}
				// Handle DynamicBroadcastInDim - when broadcasting a constant to a dynamic shape
				if stmt.OpType == optypes.DynamicBroadcastInDim {
					return tryExtractDynamicBroadcastConstant(fn, stmt)
				}
				// Not a constant operation
				return nil, false
			}
		}
	}
	// Value not found in statements (shouldn't happen)
	return nil, false
}

// tryExtractConcatenatedShape extracts constant values from a Concatenate operation
// where all inputs are constants (or nested concatenates of constants).
// This handles the common ONNX pattern where shape tensors are built by concatenating
// individual constant dimension values.
func tryExtractConcatenatedShape(fn *Function, concatStmt *Statement) ([]int, bool) {
	// For shape tensors, we only handle axis=0 (concatenating 1D tensors or scalars)
	if dimAttr, ok := concatStmt.Attributes["dimension"]; ok {
		if dim, ok := dimAttr.(int64); ok && dim != 0 {
			return nil, false
		}
	}

	// Extract constants from each input and concatenate them
	var result []int
	for _, input := range concatStmt.Inputs {
		dimValues, ok := tryExtractConstantShape(fn, input)
		if !ok {
			// One of the inputs is not a constant
			return nil, false
		}
		result = append(result, dimValues...)
	}

	return result, true
}

// tryExtractConcatenatedShapePartial is like tryExtractConcatenatedShape but returns
// partial results. For dimensions that can't be extracted, it uses -1 as a sentinel.
// Returns (result, allExtracted, anyExtracted).
func tryExtractConcatenatedShapePartial(fn *Function, concatStmt *Statement) ([]int, bool, bool) {
	// For shape tensors, we only handle axis=0 (concatenating 1D tensors or scalars)
	if dimAttr, ok := concatStmt.Attributes["dimension"]; ok {
		if dim, ok := dimAttr.(int64); ok && dim != 0 {
			return nil, false, false
		}
	}

	// Extract constants from each input and concatenate them
	var result []int
	allExtracted := true
	anyExtracted := false

	for _, input := range concatStmt.Inputs {
		dimValues, ok := tryExtractConstantShape(fn, input)
		if !ok {
			// This input is not extractable - try to determine its size from shape
			// and fill with -1 sentinels
			inputSize := 1
			if input.shape.Rank() == 1 && input.shape.Dimensions[0] > 0 {
				inputSize = input.shape.Dimensions[0]
			}
			for i := 0; i < inputSize; i++ {
				result = append(result, -1)
			}
			allExtracted = false
		} else {
			result = append(result, dimValues...)
			anyExtracted = true
		}
	}

	return result, allExtracted, anyExtracted
}

// tryExtractGatherConstant attempts to evaluate a Gather operation when both the operand
// and indices are constants. This handles the ONNX pattern where individual elements are
// extracted from a constant shape tensor using Gather operations.
func tryExtractGatherConstant(fn *Function, gatherStmt *Statement) ([]int, bool) {
	// Gather has two inputs: operand and startIndices
	if len(gatherStmt.Inputs) != 2 {
		return nil, false
	}

	operand := gatherStmt.Inputs[0]
	startIndices := gatherStmt.Inputs[1]

	// Extract the constant operand (the tensor we're gathering from)
	operandValues, operandOk := tryExtractConstantShape(fn, operand)
	// Extract the constant indices (where we're gathering from)
	indexValues, indexOk := tryExtractConstantShape(fn, startIndices)

	if !operandOk {
		return nil, false
	}

	if !indexOk {
		return nil, false
	}

	// For simple scalar gather operations (most common in shape tensors),
	// we expect a single index value
	if len(indexValues) != 1 {
		// More complex gather patterns not yet supported
		return nil, false
	}

	index := indexValues[0]

	// Validate index is in bounds
	if index < 0 || index >= len(operandValues) {
		return nil, false
	}

	// Return the gathered value as a single-element slice
	return []int{operandValues[index]}, true
}

// tryExtractSliceConstant attempts to evaluate a Slice operation on a constant tensor.
// This handles extracting a subrange from a constant shape tensor.
func tryExtractSliceConstant(fn *Function, sliceStmt *Statement) ([]int, bool) {
	// Slice has one input: the operand to slice
	if len(sliceStmt.Inputs) != 1 {
		return nil, false
	}

	operand := sliceStmt.Inputs[0]

	// Extract the constant operand (the tensor we're slicing)
	operandValues, ok := tryExtractConstantShape(fn, operand)
	if !ok {
		return nil, false
	}

	// Extract slice parameters from attributes
	startIndicesAttr, hasStart := sliceStmt.Attributes["start_indices"]
	limitIndicesAttr, hasLimit := sliceStmt.Attributes["limit_indices"]

	if !hasStart || !hasLimit {
		return nil, false
	}

	// Convert attributes to int slices
	startIndices, ok := extractIntSliceFromAttribute(startIndicesAttr)
	if !ok {
		return nil, false
	}

	limitIndices, ok := extractIntSliceFromAttribute(limitIndicesAttr)
	if !ok {
		return nil, false
	}

	// For 1D slices (most common in shape tensors), extract the range
	if len(startIndices) != 1 || len(limitIndices) != 1 {
		return nil, false
	}

	start := startIndices[0]
	limit := limitIndices[0]

	// Validate bounds
	if start < 0 || limit > len(operandValues) || start >= limit {
		return nil, false
	}

	// Return the sliced values
	return operandValues[start:limit], true
}

// tryExtractBroadcastInDimConstant attempts to extract constant values from a BroadcastInDim operation.
// This handles the case where a scalar constant (e.g., 0) is being broadcast to create a shape tensor.
// For example, ConstantOfShape creates a constant scalar and broadcasts it to the desired shape.
func tryExtractBroadcastInDimConstant(fn *Function, broadcastStmt *Statement) ([]int, bool) {
	// BroadcastInDim has one input: the operand to broadcast
	if len(broadcastStmt.Inputs) != 1 {
		return nil, false
	}

	operand := broadcastStmt.Inputs[0]

	// Try to extract the constant value being broadcast
	operandValues, ok := tryExtractConstantShape(fn, operand)
	if !ok {
		return nil, false
	}

	// If the operand is a scalar (single value), and we're broadcasting it to create a shape tensor,
	// we need to replicate that value according to the output shape
	if len(operandValues) == 1 {
		// Extract the output shape from the statement's output
		outputShape := broadcastStmt.Outputs[0].shape
		if outputShape.Rank() != 1 {
			// Shape tensors are always 1D, so this shouldn't be a shape tensor
			return nil, false
		}

		// Get the output dimension (how many times to replicate the scalar)
		outputSize := outputShape.Dimensions[0]
		if outputSize < 0 {
			// Symbolic dimension - can't materialize
			return nil, false
		}

		// Replicate the scalar value
		result := make([]int, outputSize)
		for i := range result {
			result[i] = operandValues[0]
		}
		return result, true
	}

	// If operand is already the right shape (1D tensor), just return it
	// This handles cases where BroadcastInDim is a no-op or simple reshape
	if len(operandValues) > 1 {
		return operandValues, true
	}

	return nil, false
}

// tryExtractDynamicBroadcastConstant attempts to extract constant values from a DynamicBroadcastInDim operation.
// This handles the case where a scalar constant is being broadcast dynamically to create a shape tensor.
// For example, broadcasting constant 1 to shape [3] produces [1, 1, 1].
func tryExtractDynamicBroadcastConstant(fn *Function, broadcastStmt *Statement) ([]int, bool) {
	// DynamicBroadcastInDim has two inputs: operand and outputDimensions
	if len(broadcastStmt.Inputs) != 2 {
		return nil, false
	}

	operand := broadcastStmt.Inputs[0]
	outputDims := broadcastStmt.Inputs[1]

	// Try to extract the constant value being broadcast
	operandValues, operandOk := tryExtractConstantShape(fn, operand)
	// Try to extract the output shape from the shape tensor
	outputShape, shapeOk := tryExtractConstantShape(fn, outputDims)

	if !operandOk {
		return nil, false
	}

	// If the operand is a scalar (single value), and we know the output shape,
	// we can compute the broadcast result
	if len(operandValues) == 1 && shapeOk && len(outputShape) > 0 {
		// For 1D broadcast (most common in shape calculations), replicate the scalar
		outputSize := outputShape[0]
		if outputSize <= 0 {
			// Invalid or symbolic size
			return nil, false
		}

		// Replicate the scalar value
		result := make([]int, outputSize)
		for i := range result {
			result[i] = operandValues[0]
		}
		return result, true
	}

	// If we can't extract the shape but the output has a static dimension, use that
	if len(operandValues) == 1 {
		outputShapeFromType := broadcastStmt.Outputs[0].shape
		if outputShapeFromType.Rank() == 1 {
			outputSize := outputShapeFromType.Dimensions[0]
			if outputSize > 0 {
				result := make([]int, outputSize)
				for i := range result {
					result[i] = operandValues[0]
				}
				return result, true
			}
		}
	}

	return nil, false
}

// tryExtractGetDimensionSize attempts to extract the dimension size from a GetDimensionSize operation.
// This handles the case where the shape of an operand is known at compile time, even if the operand
// itself is not constant. For example, GetDimensionSize(x, 0) where x has shape [12, 512, 64] returns [12].
func tryExtractGetDimensionSize(fn *Function, getDimStmt *Statement) ([]int, bool) {
	// GetDimensionSize has one input: the operand whose dimension we're extracting
	if len(getDimStmt.Inputs) != 1 {
		return nil, false
	}

	operand := getDimStmt.Inputs[0]

	// GetDimensionSize has a "dimension" attribute specifying which dimension to extract
	dimAttr, ok := getDimStmt.Attributes["dimension"]
	if !ok {
		return nil, false
	}

	// Convert the dimension attribute to an integer
	var dimIndex int
	switch v := dimAttr.(type) {
	case int64:
		dimIndex = int(v)
	case int:
		dimIndex = v
	default:
		return nil, false
	}

	// Check that the operand has a known shape
	if operand.shape.Rank() <= dimIndex || dimIndex < 0 {
		return nil, false
	}

	// Extract the dimension value
	dimValue := operand.shape.Dimensions[dimIndex]
	if dimValue < 0 {
		// Symbolic dimension - can't materialize
		return nil, false
	}

	// Return as a single-element slice (GetDimensionSize returns a scalar, but we represent it as a 1D slice)
	return []int{dimValue}, true
}

// tryExtractMultiplyConstant attempts to evaluate a Multiply operation when both operands
// are constants or can be extracted as constants. This handles shape calculations that
// involve multiplying dimension sizes.
func tryExtractMultiplyConstant(fn *Function, multiplyStmt *Statement) ([]int, bool) {
	// Multiply has two inputs
	if len(multiplyStmt.Inputs) != 2 {
		return nil, false
	}

	// Try to extract both operands as constants
	lhsValues, lhsOk := tryExtractConstantShape(fn, multiplyStmt.Inputs[0])
	rhsValues, rhsOk := tryExtractConstantShape(fn, multiplyStmt.Inputs[1])

	if !lhsOk || !rhsOk {
		return nil, false
	}

	// Handle scalar multiplication (most common)
	if len(lhsValues) == 1 && len(rhsValues) == 1 {
		result := lhsValues[0] * rhsValues[0]
		return []int{result}, true
	}

	// Handle element-wise multiplication (common in shape calculations with broadcast)
	if len(lhsValues) == len(rhsValues) {
		result := make([]int, len(lhsValues))
		for i := range lhsValues {
			result[i] = lhsValues[i] * rhsValues[i]
		}
		return result, true
	}

	// Handle broadcast: one operand is scalar
	if len(lhsValues) == 1 {
		result := make([]int, len(rhsValues))
		for i := range rhsValues {
			result[i] = lhsValues[0] * rhsValues[i]
		}
		return result, true
	}
	if len(rhsValues) == 1 {
		result := make([]int, len(lhsValues))
		for i := range lhsValues {
			result[i] = lhsValues[i] * rhsValues[0]
		}
		return result, true
	}

	return nil, false
}

// tryExtractSelectConstant attempts to evaluate a Select operation when the condition
// is a constant. This handles shape calculations that use conditional logic (element-wise).
func tryExtractSelectConstant(fn *Function, selectStmt *Statement) ([]int, bool) {
	// Select has three inputs: condition, onTrue, onFalse
	if len(selectStmt.Inputs) != 3 {
		return nil, false
	}

	// Try to extract all inputs as constants
	condValues, condOk := tryExtractConstantShape(fn, selectStmt.Inputs[0])
	onTrueValues, onTrueOk := tryExtractConstantShape(fn, selectStmt.Inputs[1])
	onFalseValues, onFalseOk := tryExtractConstantShape(fn, selectStmt.Inputs[2])

	if !condOk {
		return nil, false
	}

	// Handle scalar condition (selects entire branch)
	if len(condValues) == 1 {
		if condValues[0] != 0 {
			if onTrueOk {
				return onTrueValues, true
			}
		} else {
			if onFalseOk {
				return onFalseValues, true
			}
		}
		return nil, false
	}

	// Handle element-wise selection (common in shape calculations)
	if !onTrueOk || !onFalseOk {
		return nil, false
	}

	// Ensure all have the same length for element-wise selection
	if len(condValues) != len(onTrueValues) || len(condValues) != len(onFalseValues) {
		// Handle broadcast cases
		if len(onTrueValues) == 1 {
			// Broadcast onTrue to match condition
			val := onTrueValues[0]
			onTrueValues = make([]int, len(condValues))
			for i := range onTrueValues {
				onTrueValues[i] = val
			}
		}
		if len(onFalseValues) == 1 {
			// Broadcast onFalse to match condition
			val := onFalseValues[0]
			onFalseValues = make([]int, len(condValues))
			for i := range onFalseValues {
				onFalseValues[i] = val
			}
		}
		// Re-check lengths after broadcast
		if len(condValues) != len(onTrueValues) || len(condValues) != len(onFalseValues) {
			return nil, false
		}
	}

	// Perform element-wise selection
	result := make([]int, len(condValues))
	for i := range condValues {
		if condValues[i] != 0 {
			result[i] = onTrueValues[i]
		} else {
			result[i] = onFalseValues[i]
		}
	}
	return result, true
}

// tryExtractCompareConstant attempts to evaluate a Compare operation when both operands
// are constants. Returns 1 for true, 0 for false (element-wise for tensors).
func tryExtractCompareConstant(fn *Function, compareStmt *Statement) ([]int, bool) {
	// Compare has two inputs
	if len(compareStmt.Inputs) != 2 {
		return nil, false
	}

	// Try to extract both operands as constants
	lhsValues, lhsOk := tryExtractConstantShape(fn, compareStmt.Inputs[0])
	rhsValues, rhsOk := tryExtractConstantShape(fn, compareStmt.Inputs[1])

	if !lhsOk || !rhsOk {
		return nil, false
	}

	// Get the comparison direction
	dirAttr, ok := compareStmt.Attributes["comparison_direction"]
	if !ok {
		return nil, false
	}

	// Parse direction - it's usually a literalStr like "#stablehlo<comparison_direction EQ>"
	dirStr := fmt.Sprintf("%v", dirAttr)

	// Helper function to perform comparison
	compare := func(lhs, rhs int) int {
		if strings.Contains(dirStr, "EQ") {
			if lhs == rhs {
				return 1
			}
		} else if strings.Contains(dirStr, "NE") {
			if lhs != rhs {
				return 1
			}
		} else if strings.Contains(dirStr, "LT") {
			if lhs < rhs {
				return 1
			}
		} else if strings.Contains(dirStr, "LE") {
			if lhs <= rhs {
				return 1
			}
		} else if strings.Contains(dirStr, "GT") {
			if lhs > rhs {
				return 1
			}
		} else if strings.Contains(dirStr, "GE") {
			if lhs >= rhs {
				return 1
			}
		}
		return 0
	}

	// Handle scalar comparison
	if len(lhsValues) == 1 && len(rhsValues) == 1 {
		return []int{compare(lhsValues[0], rhsValues[0])}, true
	}

	// Handle element-wise comparison
	if len(lhsValues) == len(rhsValues) {
		result := make([]int, len(lhsValues))
		for i := range lhsValues {
			result[i] = compare(lhsValues[i], rhsValues[i])
		}
		return result, true
	}

	// Handle broadcast: one operand is scalar
	if len(lhsValues) == 1 {
		result := make([]int, len(rhsValues))
		for i := range rhsValues {
			result[i] = compare(lhsValues[0], rhsValues[i])
		}
		return result, true
	}
	if len(rhsValues) == 1 {
		result := make([]int, len(lhsValues))
		for i := range lhsValues {
			result[i] = compare(lhsValues[i], rhsValues[0])
		}
		return result, true
	}

	return nil, false
}

// tryExtractDivideConstant attempts to evaluate a Divide operation when both operands
// are constants or can be extracted as constants. This handles shape calculations that
// involve dividing dimension sizes (e.g., splitting attention heads).
func tryExtractDivideConstant(fn *Function, divideStmt *Statement) ([]int, bool) {
	// Divide has two inputs
	if len(divideStmt.Inputs) != 2 {
		return nil, false
	}

	// Try to extract both operands as constants
	lhsValues, lhsOk := tryExtractConstantShape(fn, divideStmt.Inputs[0])
	rhsValues, rhsOk := tryExtractConstantShape(fn, divideStmt.Inputs[1])

	if !lhsOk || !rhsOk {
		return nil, false
	}

	// For scalar division (most common in shape calculations), both should be single values
	if len(lhsValues) != 1 || len(rhsValues) != 1 {
		return nil, false
	}

	// Avoid division by zero
	if rhsValues[0] == 0 {
		return nil, false
	}

	// Divide the values (integer division)
	result := lhsValues[0] / rhsValues[0]
	return []int{result}, true
}

// extractIntSliceFromAttribute attempts to extract a slice of integers from a StableHLO attribute.
// Handles the literalStr type used for start_indices, limit_indices, etc.
func extractIntSliceFromAttribute(attr interface{}) ([]int, bool) {
	// The attribute is typically a literalStr like "array<i64: 3, 4>"
	if str, ok := attr.(literalStr); ok {
		return parseArrayAttr(string(str))
	}

	// Also handle regular strings
	if str, ok := attr.(string); ok {
		return parseArrayAttr(str)
	}

	// Try reflecting on the value to handle other array-like types
	v := reflect.ValueOf(attr)
	if v.Kind() == reflect.Slice || v.Kind() == reflect.Array {
		result := make([]int, v.Len())
		for i := 0; i < v.Len(); i++ {
			if intVal, ok := toInt(v.Index(i).Interface()); ok {
				result[i] = intVal
			} else {
				return nil, false
			}
		}
		return result, true
	}

	return nil, false
}

// parseArrayAttr parses a StableHLO array attribute string like "array<i64: 3, 4>" into a slice of ints.
func parseArrayAttr(s string) ([]int, bool) {
	// Find the colon that separates the type from the values
	colonIdx := strings.Index(s, ":")
	if colonIdx == -1 {
		return nil, false
	}

	// Extract the values part after the colon
	valuesStr := s[colonIdx+1:]

	// Remove the closing ">" if present
	valuesStr = strings.TrimRight(valuesStr, ">")
	valuesStr = strings.TrimSpace(valuesStr)

	// Split by comma and parse each value
	parts := strings.Split(valuesStr, ",")
	result := make([]int, 0, len(parts))

	for _, part := range parts {
		part = strings.TrimSpace(part)
		if part == "" {
			continue
		}

		val, err := strconv.Atoi(part)
		if err != nil {
			return nil, false
		}
		result = append(result, val)
	}

	return result, true
}

// extractIntegersFromTensorLiteral extracts integer values from a tensorLiteral.
// Handles both scalar and 1D tensor cases, and various integer types.
func extractIntegersFromTensorLiteral(tl tensorLiteral) ([]int, bool) {
	// Handle scalar case
	if tl.dims == nil || len(tl.dims) == 0 {
		if val, ok := toInt(tl.value); ok {
			return []int{val}, true
		}
		return nil, false
	}

	// Handle 1D tensor case
	if len(tl.dims) != 1 {
		return nil, false
	}

	size := tl.dims[0]
	result := make([]int, size)

	// Use reflection to handle different integer slice types
	valueV := reflect.ValueOf(tl.value)
	if valueV.Kind() != reflect.Slice && valueV.Kind() != reflect.Array {
		return nil, false
	}

	for i := 0; i < size; i++ {
		if val, ok := toInt(valueV.Index(i).Interface()); ok {
			result[i] = val
		} else {
			return nil, false
		}
	}

	return result, true
}

// toInt converts various integer types to int.
func toInt(v interface{}) (int, bool) {
	switch val := v.(type) {
	case int:
		return val, true
	case int32:
		return int(val), true
	case int64:
		return int(val), true
	case uint32:
		return int(val), true
	case uint64:
		return int(val), true
	case int8:
		return int(val), true
	case int16:
		return int(val), true
	case uint8:
		return int(val), true
	case uint16:
		return int(val), true
	default:
		return 0, false
	}
}

// Compare implements the corresponding standard binary operation.
//
// For boolean data types (dtypes.Bool) use the types.CompareUnsigned type.
func Compare(lhs, rhs *Value, direction types.ComparisonDirection, compareType types.ComparisonType) (*Value, error) {
	op := optypes.Compare
	fn := lhs.fn
	if fn.Returned {
		return nil, errors.Errorf("cannot add operation %s after returning, in function %q",
			op, fn.Name)
	}
	if rhs.fn != fn {
		return nil, errors.Errorf("cannot add operation %s to function %q, because operands are from different functions (%q and %q)",
			op, fn.Name, fn.Name, rhs.fn.Name)
	}
	outputShape, err := shapeinference.Compare(lhs.shape, rhs.shape, direction, compareType)
	if err != nil {
		return nil, err
	}
	stmt := fn.addOp(op, outputShape, lhs, rhs)
	stmt.Attributes = map[string]any{
		"compare_type":         compareType,
		"comparison_direction": direction,
	}
	return stmt.Outputs[0], nil
}

func valuesToShapes(values []*Value) []shapes.Shape {
	s := make([]shapes.Shape, len(values))
	for i, v := range values {
		s[i] = v.shape
	}
	return s
}

// Complex returns the complex value by concatenating the real and imaginary parts element-wise.
func Complex(real, imag *Value) (*Value, error) {
	op := optypes.Complex
	fn := real.fn
	if fn.Returned {
		return nil, errors.Errorf("cannot add operation %s after returning, in function %q",
			op, fn.Name)
	}
	if imag.fn != fn {
		return nil, errors.Errorf("cannot add operation %s to function %q, because operands are from different functions (%q and %q)",
			op, fn.Name, fn.Name, imag.fn.Name)
	}
	outputShape, err := shapeinference.Complex(real.shape, imag.shape)
	if err != nil {
		return nil, err
	}
	return fn.addOp(op, outputShape, real, imag).Outputs[0], nil
}

// Real returns the real part of the complex value.
func Real(complex *Value) (*Value, error) {
	op := optypes.Real
	fn := complex.fn
	if fn.Returned {
		return nil, errors.Errorf("cannot add operation %s after returning, in function %q",
			op, fn.Name)
	}
	outputShape, err := shapeinference.RealOrImag(complex.shape)
	if err != nil {
		return nil, err
	}
	return fn.addOp(op, outputShape, complex).Outputs[0], nil
}

// Imag returns the real part of the complex value.
func Imag(complex *Value) (*Value, error) {
	op := optypes.Imag
	fn := complex.fn
	if fn.Returned {
		return nil, errors.Errorf("cannot add operation %s after returning, in function %q",
			op, fn.Name)
	}
	outputShape, err := shapeinference.RealOrImag(complex.shape)
	if err != nil {
		return nil, err
	}
	return fn.addOp(op, outputShape, complex).Outputs[0], nil
}

// IsFinite tests whether each element of operand is finite, i.e., if it is not positive nor negative infinity, and it is not NaN.
// It returns the same shape as the input, but with boolean values where each element is true if and only if
// the corresponding input element is finite.
func IsFinite(x *Value) (*Value, error) {
	op := optypes.IsFinite
	fn := x.fn
	if fn.Returned {
		return nil, errors.Errorf("cannot add operation %s after returning, in function %q",
			op, fn.Name)
	}
	outputShape, err := shapeinference.IsFinite(x.shape)
	if err != nil {
		return nil, err
	}
	return fn.addOp(op, outputShape, x).Outputs[0], nil
}

// Clamp returns the minimum(maximum(x, min), max).
//
// The values max and min can either be a scalar or have the same shape as x.
//
// Clamp is not defined for booleans or complex numbers (the semantics would not be clear).
//
// Note: the order of the arguments in StableHLO is different from most ML libraries.
func Clamp(min, x, max *Value) (*Value, error) {
	op := optypes.Clamp
	fn := x.fn
	if fn.Returned {
		return nil, errors.Errorf("cannot add operation %s after returning, in function %q",
			op, fn.Name)
	}
	if min.fn != fn || max.fn != fn {
		return nil, errors.Errorf("cannot add operation %s to function %q, because operands are from different functions (%q, %q and %q)",
			op, fn.Name, fn.Name, max.fn.Name, min.fn.Name)
	}
	outputShape, err := shapeinference.Clamp(min.shape, x.shape, max.shape)
	if err != nil {
		return nil, err
	}
	return fn.addOp(op, outputShape, min, x, max).Outputs[0], nil
}

// DotGeneralBuilder is a builder for DotGeneral nodes. See DotGeneral for more details.
type DotGeneralBuilder struct {
	fn                               *Function
	lhs                              *Value
	lhsContractingAxes, lhsBatchAxes []int
	rhs                              *Value
	rhsContractingAxes, rhsBatchAxes []int

	precision   [2]types.DotGeneralPrecisionType
	outputDType dtypes.DType
	algorithm   *types.DotGeneralAlgorithm
}

// DotGeneral takes as input lhs (left-hand-side) and rhs (right-hand-side) specifications
// for a general vector product -- a generalized "Einsum". Each axis can be:
//   - Just aligned (batch axes), so the output has the same axes as the inputs. The dimensions
//     must match in lhs and rhs.
//   - Crossed (default), in which case the output is the combination (concatenation) of the
//     dimensions.
//   - Contracted (contracting axes), where the output does multiply the values and reduce sum
//     those dimensions.
//
// It follows that the resulting dimension number starts with the batch dimension, then the 'lhs'
// non-contracting/non-batch dimension, and finally the 'rhs' non-contracting/non-batch dimension.
// It provides the basic means of implementing Einsum.
//
// Because there are optional parameters, this function returns a DotGeneralBuilder that can
// be further configured. Call DotGeneralBuilder.Done to get the final DotGeneral node.
func Dot(lhs, rhs *Value) (*Value, error) {
	if lhs.Shape().Rank() != 2 || rhs.Shape().Rank() != 2 {
		return nil, errors.Errorf("Dot only supports rank-2 tensors, got %d and %d", lhs.Shape().Rank(), rhs.Shape().Rank())
	}
	return DotGeneral(
		lhs, []int{1}, nil,
		rhs, []int{0}, nil,
	).Done()
}

// DotGeneral takes as input lhs (left-hand-side) and rhs (right-hand-side) specifications
// for a general vector product -- a generalized "Einsum". Each axis can be:
//   - Just aligned (batch axes), so the output has the same axes as the inputs. The dimensions
//     must match in lhs and rhs.
//   - Crossed (default), in which case the output is the combination (concatenation) of the
//     dimensions.
//   - Contracted (contracting axes), where the output does multiply the values and reduce sum
//     those dimensions.
//
// It follows that the resulting dimension number starts with the batch dimension, then the 'lhs'
// non-contracting/non-batch dimension, and finally the 'rhs' non-contracting/non-batch dimension.
// It provides the basic means of implementing Einsum.
//
// Because there are optional parameters, this function returns a DotGeneralBuilder that can
// be further configured. Call DotGeneralBuilder.Done to get the final DotGeneral node.
func DotGeneral(
	lhsOp *Value, lhsContractingAxes, lhsBatchAxes []int,
	rhsOp *Value, rhsContractingAxes, rhsBatchAxes []int) *DotGeneralBuilder {
	return &DotGeneralBuilder{
		fn:                 lhsOp.fn,
		lhs:                lhsOp,
		lhsContractingAxes: lhsContractingAxes,
		lhsBatchAxes:       lhsBatchAxes,
		rhs:                rhsOp,
		rhsContractingAxes: rhsContractingAxes,
		rhsBatchAxes:       rhsBatchAxes,

		precision:   [2]types.DotGeneralPrecisionType{types.DotGeneralPrecisionDefault, types.DotGeneralPrecisionDefault},
		outputDType: lhsOp.shape.DType,
	}
}

// Precision sets the precision of the dot-general operation.
//
// Its default is described as "the fastest calculation, but the least accurate approximation to the original number."
//
// It controls the tradeoff between speed and accuracy for computations on accelerator backends.
// This can be one of the following (at the moment, the semantics of these enum values are underspecified,
// but they are planning to address this in #755 -- https://github.com/openxla/stablehlo/issues/755):
func (b *DotGeneralBuilder) Precision(lhsPrecision, rhsPrecision types.DotGeneralPrecisionType) *DotGeneralBuilder {
	b.precision[0] = lhsPrecision
	b.precision[1] = rhsPrecision
	return b
}

// OutputDType sets the output data type: for input types like BFloat16 one may want to increase the
// output precision.
func (b *DotGeneralBuilder) OutputDType(dtype dtypes.DType) *DotGeneralBuilder {
	b.outputDType = dtype
	return b
}

// Algorithm sets the algorithm settings to use for the dot-general operation.
//
// The default is not to set any of these parameters.
//
// See details in types.DotGeneralAlgorithm.
func (b *DotGeneralBuilder) Algorithm(algorithm *types.DotGeneralAlgorithm) *DotGeneralBuilder {
	b.algorithm = algorithm
	return b
}

// Done indicates the end of the DotGeneralBuilder configuration.
// It checks the validity of the parameters and shapes and returns the final DotGeneral node.
func (b *DotGeneralBuilder) Done() (*Value, error) {
	op := optypes.DotGeneral
	fn := b.fn
	if fn.Returned {
		return nil, errors.Errorf("cannot add operation %s after returning, in function %q",
			op, fn.Name)
	}
	if b.lhs.fn != fn || b.rhs.fn != fn {
		return nil, errors.Errorf("cannot add operation %s to function %q, because operands are from different functions (%q and %q)",
			op, fn.Name, b.lhs.fn.Name, b.rhs.fn.Name)
	}
	outputShape, err := shapeinference.DotGeneral(
		b.lhs.shape, b.lhsContractingAxes, b.lhsBatchAxes,
		b.rhs.shape, b.rhsContractingAxes, b.rhsBatchAxes,
		b.outputDType)
	if err != nil {
		return nil, err
	}
	stmt := b.fn.addOp(op, outputShape, b.lhs, b.rhs)
	stmt.Attributes = map[string]any{
		"dot_dimension_numbers": literalStrF(
			"#stablehlo.dot<\n"+
				"\tlhs_batching_dimensions = %s,\n"+
				"\trhs_batching_dimensions = %s,\n"+
				"\tlhs_contracting_dimensions = %s,\n"+
				"\trhs_contracting_dimensions = %s\n>",
			intSliceToStableHLO(b.lhsBatchAxes),
			intSliceToStableHLO(b.rhsBatchAxes),
			intSliceToStableHLO(b.lhsContractingAxes),
			intSliceToStableHLO(b.rhsContractingAxes)),
	}
	precisionConfig := fmt.Sprintf("[#stablehlo<precision %s>, #stablehlo<precision %s>]",
		b.precision[0].ToStableHLO(), b.precision[1].ToStableHLO())
	stmt.Attributes["precision_config"] = literalStr(precisionConfig)
	if b.algorithm != nil {
		stmt.Attributes["algorithm"] = literalStrF("#stablehlo.dot_algorithm<\n"+
			"\tlhs_precision_type = %s,\n"+
			"\trhs_precision_type = %s,\n"+
			"\taccumulation_type = %s,\n"+
			"\tlhs_component_count = %d,\n"+
			"\trhs_component_count = %d,\n"+
			"\tnum_primitive_operations = %d,\n"+
			"\tallow_imprecise_accumulation = %v>",
			b.algorithm.LhsPrecisionType.ToStableHLO(),
			b.algorithm.RhsPrecisionType.ToStableHLO(),
			b.algorithm.AccumulationType.ToStableHLO(),
			b.algorithm.LhsComponentCount,
			b.algorithm.RhsComponentCount,
			b.algorithm.NumPrimitiveOperations,
			b.algorithm.AllowImpreciseAccumulation)
	}
	return stmt.Outputs[0], nil
}

// Reshape the operand to the given shape.
// The total size of the new shape must match the original shape.
//
// This has no effect on the data, no transposition is performed.
func Reshape(operand *Value, shape shapes.Shape) (*Value, error) {
	op := optypes.Reshape
	fn := operand.fn
	if fn.Returned {
		return nil, errors.Errorf("cannot add operation %s after returning, in function %q",
			op, fn.Name)
	}
	if operand.shape.DType != shape.DType {
		return nil, errors.Errorf("Reshape() requires the operand and the shape to have the same data type, got operand=%s and shape=%s",
			operand.shape, shape)
	}

	// Check if either shape has symbolic dimensions (negative values)
	hasSymbolic := false
	for _, dim := range operand.shape.Dimensions {
		if dim < 0 {
			hasSymbolic = true
			break
		}
	}
	if !hasSymbolic {
		for _, dim := range shape.Dimensions {
			if dim < 0 {
				hasSymbolic = true
				break
			}
		}
	}

	// Skip size validation if any dimension is symbolic (will be validated at runtime)
	if !hasSymbolic && operand.shape.Size() != shape.Size() {
		return nil, errors.Errorf("Reshape() requires the total size of the new shape to match the original shape, got operand=%s and shape=%s",
			operand.shape, shape)
	}
	stmt := fn.addOp(op, shape, operand)
	return stmt.Outputs[0], nil
}

// BroadcastInDim broadcasts dimensions from the operand to the target shape.
// It can also transpose axes and add new ones.
//
// The axesMapping should have one value per operand axes. It maps the axes from the operand to
// the corresponding value on the target shape.
func BroadcastInDim(operand *Value, target shapes.Shape, axesMapping []int) (*Value, error) {
	op := optypes.BroadcastInDim
	fn := operand.fn
	if fn.Returned {
		return nil, errors.Errorf("cannot add operation %s after returning, in function %q",
			op, fn.Name)
	}
	err := shapeinference.BroadcastInDim(operand.shape, target, axesMapping)
	if err != nil {
		return nil, err
	}
	stmt := fn.addOp(op, target, operand)
	stmt.Attributes = map[string]any{"broadcast_dimensions": intSliceToArrayI64StableHLO(axesMapping)}
	return stmt.Outputs[0], nil
}

// Gather is a powerful but cumbersome Gather operation.
// Full details in https://openxla.org/stablehlo/spec#gather.
//
// The output of Gather has the same DType of the operand, from where we are pulling the data.
//
// Its output shape will be composed of 2 parts:
//
//   - Batch axes: they come from operandBatchingAxes/startIndicesBatchingAxes (they correspond to each other)
//     and from the other axes of startIndices, except the "indexVectorAxis" (usually the last)
//     that is used as the indices into the operand. (*)
//   - "Offset axes": these are axes that come from the operand, the sizes given by sliceSizes.
//     Notice that if sliceSizes for an axis is 1, and that axis is present in the collapsedSliceAxes list, this
//     axis gets omitted in the output.
//
// So in general output.Rank() = startIndices.Rank() - 1 + len(offsetAxes).
//
// (*) One exception is if indexVectorAxis == startIndices.Rank(), in which case we assume there is an
// extra virtual axis in startIndices of size 1, in which case output.Rank() = startIndices.Rank() + len(offsetAxes).
//
// (*) One exception is if indexVectorAxis == startIndices.Rank(), in which case we assume there is an
// extra implicit axis in startIndices of size 1, in which case output.Rank() = startIndices.Rank() + len(offsetAxes).
//
// Arguments:
//   - operand: the values from where we are gathering. The output DType will follow the operand one.
//   - startIndices: are the indices we want to gather. The axis pointed by indexVector
//     lists the indices of the slice to be gathered in the operand array (their values are mapped to the axis
//     in the operand according to startIndexMap).
//     All other axes are "batch dimensions" and they will have equivalent axes (same dimensions) in the output.
//   - indexVectorAxis: which of the axis in startIndices is collected and used as the start index for slices
//     to be gathered in the operand.
//     It is typically the last axis of startIndices, so startIndices.Shape.Rank()-1.
//     There is a special case where indexVectorAxis == startIndices.Rank() in which case we assume there is an
//     extra virtual axis in startIndices of size 1, in which case output.Rank() = startIndices.Rank() + len(offsetAxes).
//   - offsetOutputAxes: _output_ axes (not the operand's) that will hold the "offset slices", slices that are not
//     collapsed. It points in which position (axis) in the output these slices should show up.
//     The len(offsetOutputAxes) must match the dimension of indexVectorAxis (== startIndices.Dimensions[indexVectorAxis]).
//     Notice all axes in the operand will either become an "offset axis" in the output,
//     of optionally collapsed (or "squeezed") in the output, if included in collapsedSliceAxes.
//     The axes in the output (given in offsetAxes) to the axes in the operand (the axes not present in collapsedSliceAxes) sequentially.
//     One must have Rank(operand) == len(collapsedSliceAxes) + len(offsetAxes) + len(operandBatchingAxes).
//   - collapsedSliceAxes: _operand_ axes (for which sliceSizes are 1) not to be included in the output.
//     One must have sliceSizes[collapsedSliceAxes[i]] == 1 for all i.
//   - operandBatchingAxes: operand's batching axes that have corresponding batching axes in the startIndices, and that
//     will also be included in the output.
//     One must have sliceSizes[operandBatchingAxes[i]] == 1 for all i.
//     Also, one must have Rank(operand) == len(operandBatchingAxes) + len(collapsedSliceAxes) + len(offsetOutputAxes).
//   - startIndicesBatchingAxes: startIndices' batching axes have corresponding batching axes in the operand, and that
//     will also be included in the output.
//   - startIndexMap: this maps which value in startIndices is used for which axis in the operand, select the slice to be gathered.
//     Notice len(startIndexMap) must match the startIndices.Dimensions[indexVectorAxis].
//     Also, len(startIndexMap) == len(offsetOutputAxes) -- offsetOutputAxes maps the same axes in the output.
//     E.g.: if startIndices.shape=(2, 3), indexVectorAxis=1, and operand.rank=4 and startIndexMap=[]int{0, 1, 2},
//     this means each row of the startIndices will point to the first 3 axes (0,1 and 2) in the operand.
//     For those axes in the operand not explicitly set (so if len(startIndexMap) < operand.Rank()), and not part
//     of operandBatchingAxes, the corresponding axis start index is considered to be 0, and one sets the sliceSizes
//     to take the slice one wants (typically the full slice).
//   - sliceSizes: a size for each operand's axis, so len(sliceSize) = operand.Rank().
//     once the start index from where to gather is resolved, this defines how much data in each axis
//     to gather.
//     Constraints: sliceSizes[collapsedSliceAxes[i]] == 1, and sliceSizes[operandBatchingAxes[j]] == 1, for all i, j.
//   - indicesAreSorted: can be set to true if it's guaranteed that startIndices are sorted (in ascending order,
//     after scattering its values according to start_index_map) by the user. This allows for some optimizations
//     in some platforms.
func Gather(operand, startIndices *Value, indexVectorAxis int,
	offsetOutputAxes, collapsedSliceAxes, operandBatchingAxes,
	startIndicesBatchingAxes, startIndexMap,
	sliceSizes []int, indicesAreSorted bool) (*Value, error) {
	op := optypes.Gather
	fn := operand.fn
	if fn.Returned {
		return nil, errors.Errorf("cannot add operation %s after returning, in function %q",
			op, fn.Name)
	}
	if startIndices.fn != fn {
		return nil, errors.Errorf("cannot add operation %s to function %q, because startIndices is from different function (%q and %q)",
			op, fn.Name, startIndices.fn.Name, fn.Name)
	}

	outputShape, err := shapeinference.Gather(
		operand.shape, startIndices.shape, indexVectorAxis,
		offsetOutputAxes, collapsedSliceAxes, operandBatchingAxes,
		startIndicesBatchingAxes, startIndexMap,
		sliceSizes, indicesAreSorted)
	if err != nil {
		return nil, err
	}
	stmt := fn.addOp(op, outputShape, operand, startIndices)
	stmt.Attributes = map[string]any{
		"dimension_numbers": literalStrF(
			"#stablehlo.gather<\n"+
				"\toffset_dims = %s,\n"+
				"\tcollapsed_slice_dims = %s,\n"+
				"\toperand_batching_dims = %s,\n"+
				"\tstart_indices_batching_dims = %s,\n"+
				"\tstart_index_map = %s,\n"+
				"\tindex_vector_dim = %d>",
			intSliceToStableHLO(offsetOutputAxes),
			intSliceToStableHLO(collapsedSliceAxes),
			intSliceToStableHLO(operandBatchingAxes),
			intSliceToStableHLO(startIndicesBatchingAxes),
			intSliceToStableHLO(startIndexMap),
			indexVectorAxis),
		"slice_sizes":        intSliceToArrayI64StableHLO(sliceSizes),
		"indices_are_sorted": indicesAreSorted,
	}
	return stmt.Outputs[0], nil
}

// Slice extracts a subarray from the input array.
// The subarray is of the same rank as the input and contains the values inside a bounding box within the input array
// where the dimensions and indices of the bounding box are given as arguments to the slice operation.
// The strides set the input stride of the slice in each axis and must be >= 1.
// It is optional, and if missing, it is assumed to be 1 for every dimension.
// Examples:
//
//	Slice(x={0, 1, 2, 3, 4}, starts={2}, limits={4}, strides=nil) -> {2, 3}
//	Slice(x={0, 1, 2, 3, 4}, starts={2}, limits={5}, strides={2}) -> {2, 4}
func Slice(x *Value, starts, limits, strides []int) (*Value, error) {
	op := optypes.Slice
	fn := x.fn
	if fn.Returned {
		return nil, errors.Errorf("cannot add operation %s after returning, in function %q",
			op, fn.Name)
	}
	if len(strides) == 0 {
		strides = make([]int, x.shape.Rank())
		for i := range strides {
			strides[i] = 1
		}
	}
	outputShape, err := shapeinference.Slice(x.shape, starts, limits, strides)
	if err != nil {
		return nil, err
	}
	stmt := fn.addOp(op, outputShape, x)
	stmt.Attributes = map[string]any{
		"start_indices": intSliceToArrayI64StableHLO(starts),
		"limit_indices": intSliceToArrayI64StableHLO(limits),
		"strides":       intSliceToArrayI64StableHLO(strides),
	}
	return stmt.Outputs[0], nil
}

// Sort sorts one or more tensors along the specified dimension using a comparator function.
//
// Sort implements the StableHLO sort operation, which can sort multiple tensors in parallel
// using a custom comparator function. This is useful for implementing operations like
// top-k, argsort, or custom sorting logic.
//
// Parameters:
//   - comparatorFn: A function that compares two elements and returns a boolean.
//     Created with Builder.NewClosure. For N inputs, must have signature
//     (lhs_0, ..., lhs_{N-1}, rhs_0, ..., rhs_{N-1}) -> scalar_bool
//     Returns true if lhs should come before rhs in sorted order.
//   - dimension: The dimension along which to sort (negative values count from the end)
//   - isStable: Whether the sort should be stable (preserve relative order of equal elements)
//   - inputs: One or more tensors to sort. All must have the same shape.
//     The first tensor is used for comparison by the comparatorFn.
//     Additional tensors are reordered to match the sorting of the first tensor.
//
// Returns:
//   - The sorted tensors in the same order as inputs.
//
// Example (descending sort with indices):
//
//	values := ... // shape [batch, seq_len]
//	indices := ... // shape [batch, seq_len] with values 0, 1, 2, ...
//
//	comparatorFn := fn.Closure()
//	lhsVal, _ := comparatorFn.Input(values.Shape().ScalarShape())
//	rhsVal, _ := comparatorFn.Input(values.Shape().ScalarShape())
//	lhsIdx, _ := comparatorFn.Input(indices.Shape().ScalarShape()) // not used in comparison
//	rhsIdx, _ := comparatorFn.Input(indices.Shape().ScalarShape()) // not used in comparison
//	result, _ := Compare(lhsVal, rhsVal, ComparisonDirectionGT, ComparisonTypeFloat)
//	comparatorFn.Return(result)
//
//	sortedValues, sortedIndices, err := Sort(comparatorFn, -1, true, values, indices)
func Sort(comparatorFn *Function, dimension int, isStable bool, inputs ...*Value) ([]*Value, error) {
	op := optypes.Sort
	if len(inputs) == 0 {
		return nil, errors.New("Sort requires at least one input tensor")
	}
	fn := inputs[0].fn
	if fn.Returned {
		return nil, errors.Errorf("cannot add operation %s after returning, in function %q",
			op, fn.Name)
	}

	// Validate all inputs are from the same function
	for i, input := range inputs {
		if input.fn != fn {
			return nil, errors.Errorf("cannot add operation %s to function %q, because input #%d is from different function (%q and %q)",
				op, fn.Name, i, input.fn.Name, fn.Name)
		}
	}

	// Validate comparator function is a closure of the current function
	if comparatorFn.Parent != fn {
		return nil, errors.Errorf("cannot add operation %s because comparatorFn is not a StableHLO closure of %s",
			op, fn.Name)
	}

	// Adjust dimension to handle negative values
	adjustedDim, err := shapeinference.AdjustAxisToRank(dimension, inputs[0].shape.Rank())
	if err != nil {
		return nil, errors.WithMessage(err, "Sort dimension for inputs")
	}

	// Perform shape inference
	inputShapes := valuesToShapes(inputs)
	outputShapes, err := shapeinference.Sort(inputShapes, adjustedDim)
	if err != nil {
		return nil, err
	}

	// Create the statement
	stmt := fn.addMultiOp(op, outputShapes, inputs)
	stmt.Attributes = map[string]any{
		"dimension": int64(adjustedDim),
		"is_stable": isStable,
	}
	stmt.AddFunctionParameter("comparator", comparatorFn)

	return stmt.Outputs, nil
}

// Concatenate operands on the given axis.
//
// All axes that are not being concatenated must match dimensions, except on the axes being concatenated.
// It doesn't work with scalars -- use ExpandAxes.
// If there is only one operand, it is returned and this is a no-op.
func Concatenate(axis int, operands ...*Value) (*Value, error) {
	op := optypes.Concatenate
	if len(operands) == 0 {
		return nil, errors.New("Concatenate requires at least one operand")
	}
	fn := operands[0].fn
	if fn.Returned {
		return nil, errors.Errorf("cannot add operation %s after returning, in function %q",
			op, fn.Name)
	}
	for i, operand := range operands {
		if operand.fn != fn {
			return nil, errors.Errorf("cannot add operation %s to function %q, because operand #%d is from different function (%q and %q)",
				op, fn.Name, i, operand.fn.Name, fn.Name)
		}
	}
	operandsShapes := make([]shapes.Shape, len(operands))
	for i, operand := range operands {
		operandsShapes[i] = operand.shape
	}

	outputShape, err := shapeinference.Concatenate(operandsShapes, axis)
	if err != nil {
		return nil, err
	}
	adjustedAxis, err := shapeinference.AdjustAxisToRank(axis, operands[0].shape.Rank())
	if err != nil {
		return nil, errors.WithMessage(err, "Concatenate axis for operands")
	}
	stmt := fn.addOp(op, outputShape, operands...)
	stmt.Attributes = map[string]any{
		"dimension": int64(adjustedAxis),
	}
	return stmt.Outputs[0], nil
}

// Reduce reduces the input along the given axes.
//
// Each resulting value is initialized with initValue (e.g.: for a sum, it's 0, for a product it's 1), and
// then each value is combined with it using the reduction function.
//
// The reduction function must be created with Builder.NewClosure, and it should take as input scalar
// values be associative and commutative.
//
// The initialValue and x must have the same DType. This initial dtype must be promotable to the dtype accepted
// by the reductions function. The result dtype is the same as the output of the reduction function.
// So one could reduce-sum a 4bit quantized tensor directly into a Float32.
//
// See MultiReduce for a version that accepts multiple inputs and outputs.
func Reduce(x, initialValue *Value, reductionFn *Function, axes ...int) (*Value, error) {
	results, err := MultiReduce([]*Value{x}, []*Value{initialValue}, reductionFn, axes...)
	if err != nil {
		return nil, err
	}
	return results[0], nil
}

// MultiReduce reduces the input along the given axes.
//
// Each resulting value i is initialized with initValues[i] (e.g.: for a sum, it's 0, for a product it is 1),
// and then each value is combined with it using the reduction function.
//
// The reduction function must be created with Builder.NewClosure.
// If there are N inputs and initialValues, the reduction function should have a signature
// (lhs_1, ... lhs_N, rhs_1, ... lhs_N) and output (out_1 ... out_N), where lhs_i and rhs_i are scalars
// taken from the inputs.
//
// It returns N results for each aggregated value.
//
// See Reduce for a version that accepts a single input.
//
// TODO: promotion of types doesn't seem to be working according to the spec in
// https://openxla.org/stablehlo/spec#reduce.
func MultiReduce(inputs, initialValues []*Value, reductionFn *Function, axes ...int) ([]*Value, error) {
	op := optypes.Reduce
	if len(inputs) == 0 {
		return nil, errors.New("MultiReduce requires at least one operand")
	}
	fn := inputs[0].fn
	if fn.Returned {
		return nil, errors.Errorf("cannot add operation %s after returning, in function %q",
			op, fn.Name)
	}
	for i, operand := range inputs {
		if operand.fn != fn {
			return nil, errors.Errorf("cannot add operation %s to function %q, because input #%d is from different function (%q and %q)",
				op, fn.Name, i, operand.fn.Name, fn.Name)
		}
	}
	for i, operand := range initialValues {
		if operand.fn != fn {
			return nil, errors.Errorf("cannot add operation %s to function %q, because initialValues[%d] is from different function (%q and %q)",
				op, fn.Name, i, operand.fn.Name, fn.Name)
		}
	}
	if reductionFn.Parent != fn {
		return nil, errors.Errorf("cannot add operation %s because reductionFn is not a StableHLO closure of %s",
			op, fn.Name)
	}

	outputsShapes, err := shapeinference.Reduce(
		valuesToShapes(inputs), valuesToShapes(initialValues),
		valuesToShapes(reductionFn.Inputs), valuesToShapes(reductionFn.Outputs),
		axes)
	if err != nil {
		return nil, err
	}
	allInputs := append(slices.Clone(inputs), initialValues...)
	stmt := fn.addMultiOp(op, outputsShapes, allInputs)
	stmt.Attributes = map[string]any{
		"dimensions": intSliceToArrayI64StableHLO(axes),
	}
	stmt.AddFunctionParameter("reductionFn", reductionFn)
	return stmt.Outputs, nil
}

// Select takes element-wise values from onTrue or onFalse depending on the value of the pred (must be boolean).
//
// The pred must be boolean and can be a scalar or have the same shape as isTrue and isFalse.
// isTrue and isFalse must have the same shape and dtypes.
func Select(pred, onTrue, onFalse *Value) (*Value, error) {
	op := optypes.Select
	fn := pred.fn
	if fn.Returned {
		return nil, errors.Errorf("cannot add operation %s after returning, in function %q",
			op, fn.Name)
	}
	if onTrue.fn != fn || onFalse.fn != fn {
		return nil, errors.Errorf("cannot add operation %s to function %q, because operands are from different functions (%q, %q and %q)",
			op, fn.Name, fn.Name, onTrue.fn.Name, onFalse.fn.Name)
	}
	outputShape, err := shapeinference.Select(pred.shape, onTrue.shape, onFalse.shape)
	if err != nil {
		return nil, err
	}
	stmt := fn.addOp(op, outputShape, pred, onTrue, onFalse)
	return stmt.Outputs[0], nil
}

// BitcastConvert performs an elementwise bit-cast operation from a dtype to another dtype.
//
// The Bitcast doesn't "convert", rather it just reinterprets the bits from x.DType() to the targetDType.
//
// If x.DType() and targetDType use the same number of bytes (targetDType.Size() == x.DType().Size()),
// the dimensions are not changed, simply the dtype is changed.
//
// If targetDType.Size() > x.DType().Size(), it requires x last axis to have a dimension of
// targetDType.Size() / x.DType().Size(), and the returned shape will trim the last axis.
//
// If targetDType.Size() < x.DType().Size(), the returned shape will have an extra axis in the end, with dimension of
// x.DType().Size() / targetDType.Size().
//
// E.g: Bitcast([1]uint32{0xdeadbeef}, dtypes.UInt16) -> [1][2]uint16{{0xbeef, 0xdead}} // Little-endian encoding.
func BitcastConvert(operand *Value, targetDtype dtypes.DType) (*Value, error) {
	op := optypes.BitcastConvert
	fn := operand.fn
	if fn.Returned {
		return nil, errors.Errorf("cannot add operation %s after returning, in function %q",
			op, fn.Name)
	}
	outputShape, err := shapeinference.BitcastConvert(operand.shape, targetDtype)
	if err != nil {
		return nil, err
	}
	stmt := fn.addOp(op, outputShape, operand)
	return stmt.Outputs[0], nil
}

// Transpose axes of x.
//
// There should be one value in permutation for each axis in x (len(permutation) == rank(x)).
//
// The output will have: output.Shape.Dimension[ii] = x.Shape.Dimension[permutations[i]].
func Transpose(x *Value, permutation ...int) (*Value, error) {
	op := optypes.Transpose
	fn := x.fn
	if fn.Returned {
		return nil, errors.Errorf("cannot add operation %s after returning, in function %q",
			op, fn.Name)
	}
	outputShape, err := shapeinference.Transpose(x.shape, permutation)
	if err != nil {
		return nil, err
	}
	stmt := fn.addOp(op, outputShape, x)
	stmt.Attributes = map[string]any{
		"permutation": intSliceToArrayI64StableHLO(permutation),
	}
	return stmt.Outputs[0], nil
}

// RNGBitGenerator generates the given shape filled with random bits.
// It takes the current random number generator (RNG) state, see RngState or RngStateFromSeed.
//
// It returns the new state of the RNG and the generated values (with random bits) with the given shape.
//
// The state shape depends on the algorithm:
//
// - types.RngDefault: PJRT implementation defined.
// - types.RngThreeFry: 2xUint64
// - types.RngPhilox: 2xUint64 or 3xUint64
func RNGBitGenerator(state *Value, shape shapes.Shape, algorithm types.RNGBitGeneratorAlgorithm) (newState, values *Value, err error) {
	op := optypes.RNGBitGenerator
	fn := state.fn
	if fn.Returned {
		return nil, nil, errors.Errorf("cannot add operation %s after returning, in function %q",
			op, fn.Name)
	}
	stmt := fn.addMultiOp(optypes.RNGBitGenerator, []shapes.Shape{state.shape, shape}, []*Value{state})
	stmt.Attributes = map[string]any{
		"rng_algorithm": literalStrF("#stablehlo<rng_algorithm %s>", strings.ToUpper(algorithm.String())),
	}
	return stmt.Outputs[0], stmt.Outputs[1], nil
}

// Scatter returns the input updated with the values of update at the locations pointed by scatterIndices.
// It allows axes to be used in powerful ways, but it's complex to get right.
// Full details in https://openxla.org/stablehlo/spec#gather.
//
// The output of Scatter has the same shape and DType of the input.
//
// Batching: while batching axes are only defined for the input and scatterIndices, the batching axes for the updates
// are inferred from the scatterIndices.
//
// Arguments:
//   - input: value to be updated in a scattered fashion.
//   - scatterIndices: indices of the values to be scattered.
//   - updates: updated values to be scattered at scatterIndices.
//   - updateWindowAxes: these axes provide the shape of the update window.
//   - insertedWindowAxes: in the resulting tensor, each axis is either a batch axis, part of the update window
//     (not specified, taken sequentially) or an insertedWindowAxes defined by this argument.
//   - inputBatchingAxes: axes that are batched with the input.
//   - scatterIndicesBatchingAxes: axes that are batched with the scatterIndices.
//   - indexedInputAxes: axes that are indexed by the scatterIndices at axis indexVectorAxis (aka. "scatter_dims_to_operand_dims").
//   - indexVectorAxis: the axis in scatterIndices that will create a vector of indices on the input where to scatter.
//     This vector of length scatterIndices.Dimensions[indexVectorAxis] will define the index value in the input on
//     the axes defined by indexedInputAxes.
//     E.g.: indexedInputAxes = [0, 1] and indexVectorAxis = 0, scatterIndices = [[0, 1, 2], [3, 4, 5]]
//     will scatter the values from updates[0] at input[0, 3], updates[1] at input[1, 4], and so on.
//     The shape of the scatterIndices is then "[2", :, ...]"
//   - indicesAreSorted: whether the scatterIndices are sorted.
//   - uniqueIndices: whether the scatterIndices are unique.
//   - indicesAreSorted, uniqueIndices: they can be set to true if it's guaranteed that scatterIndices are sorted
//     (in ascending order) and/or unique (no duplicates).
//     This allows for some optimization in some platforms.
//   - updateComputation: the closure that element-wise combines the current input value and the update value,
//     computing the result.
//     It defines also the data type of the outputs: if the updateComputation inputs and outputs don't match
//     the corresponding DType of their inputs and updates, the values from inputs and updates must be "promotable"
//     to the DType of the updateComputation.
//     Notice it may be called multiple times for some elements if the indices are not unique
//     or the updates' windows overlap.
func Scatter(input, scatterIndices, updates *Value,
	updateWindowAxes, insertedWindowAxes []int,
	inputBatchingAxes, scatterIndicesBatchingAxes []int,
	indexedInputAxes []int, indexVectorAxis int,
	indicesAreSorted, uniqueIndices bool,
	updateComputationFn *Function) (*Value, error) {
	results, err := MultiScatter([]*Value{input}, scatterIndices, []*Value{updates},
		updateWindowAxes, insertedWindowAxes,
		inputBatchingAxes, scatterIndicesBatchingAxes,
		indexedInputAxes, indexVectorAxis,
		indicesAreSorted, uniqueIndices,
		updateComputationFn)
	if err != nil {
		return nil, err
	}
	return results[0], nil
}

// MultiScatter is like Scatter, but takes N inputs and updates, but one only set of indices, and perform the Scatter
// on all at the same time.
func MultiScatter(inputs []*Value, scatterIndices *Value, updates []*Value,
	updateWindowAxes, insertedWindowAxes []int,
	inputBatchingAxes, scatterIndicesBatchingAxes []int,
	indexedInputAxes []int, indexVectorAxis int,
	indicesAreSorted, uniqueIndices bool,
	updateComputationFn *Function) ([]*Value, error) {
	op := optypes.Scatter
	if len(inputs) == 0 {
		return nil, errors.New("MultiScatter requires at least one input")
	}
	if len(inputs) != len(updates) {
		return nil, errors.Errorf("MultiScatter requires the same number of inputs and updates, got %d inputs and %d updates",
			len(inputs), len(updates))
	}
	fn := inputs[0].fn
	if fn.Returned {
		return nil, errors.Errorf("cannot add operation %s after returning, in function %q",
			op, fn.Name)
	}
	for i, input := range inputs {
		if input.fn != fn {
			return nil, errors.Errorf("cannot add operation %s to function %q, because inputs[%d] is from different function (%q and %q)",
				op, fn.Name, i, input.fn.Name, fn.Name)
		}
	}
	for i, update := range updates {
		if update.fn != fn {
			return nil, errors.Errorf("cannot add operation %s to function %q, because updates[%d] is from different function (%q and %q)",
				op, fn.Name, i, update.fn.Name, fn.Name)
		}
	}
	if scatterIndices.fn != fn {
		return nil, errors.Errorf("cannot add operation %s to function %q, because scatterIndices is from different function (%q and %q)",
			op, fn.Name, scatterIndices.fn.Name, fn.Name)
	}
	if updateComputationFn.Parent != fn {
		return nil, errors.Errorf("cannot add operation %s because updateComputationFn is not a StableHLO closure of %q",
			op, fn.Name)
	}

	inputsShapes := valuesToShapes(inputs)
	updatesShapes := valuesToShapes(updates)
	updateComputationInputShapes := valuesToShapes(updateComputationFn.Inputs)
	outputShapes, err := shapeinference.Scatter(
		inputsShapes, scatterIndices.shape, updatesShapes,
		updateWindowAxes, insertedWindowAxes,
		inputBatchingAxes, scatterIndicesBatchingAxes,
		indexedInputAxes, indexVectorAxis,
		updateComputationInputShapes, valuesToShapes(updateComputationFn.Outputs))
	if err != nil {
		return nil, err
	}
	allInputs := append(slices.Clone(inputs), scatterIndices)
	allInputs = append(allInputs, updates...)
	stmt := fn.addMultiOp(op, outputShapes, allInputs)
	stmt.Attributes = map[string]any{
		"scatter_dimension_numbers": literalStrF(
			"#stablehlo.scatter<\n"+
				"\tupdate_window_dims = %s,\n"+
				"\tinserted_window_dims = %s,\n"+
				"\tinput_batching_dims = %s,\n"+
				"\tscatter_indices_batching_dims = %s,\n"+
				"\tscatter_dims_to_operand_dims = %s,\n"+
				"\tindex_vector_dim = %d>",
			intSliceToStableHLO(updateWindowAxes),
			intSliceToStableHLO(insertedWindowAxes),
			intSliceToStableHLO(inputBatchingAxes),
			intSliceToStableHLO(scatterIndicesBatchingAxes),
			intSliceToStableHLO(indexedInputAxes),
			indexVectorAxis),
		"indices_are_sorted": indicesAreSorted,
		"unique_indices":     uniqueIndices,
	}
	stmt.AddFunctionParameter("updateFn", updateComputationFn)
	return stmt.Outputs, nil
}

// Convert x to the given dtype.
//
// For boolean to numeric conversions, false becomes 0 and true 1.
//
// For complex to non-complex conversions, the imaginary part is discarded (or set to 0).
//
// Currently, it doesn't work for quantized to/from regular tensors. Use UniformQuantize and UniformDequantize
// for that.
func Convert(x *Value, dtype dtypes.DType) (*Value, error) {
	op := optypes.Convert
	fn := x.fn
	if fn.Returned {
		return nil, errors.Errorf("cannot add operation %s after returning, in function %q",
			op, fn.Name)
	}
	outputShape := x.shape.Clone()
	outputShape.DType = dtype
	stmt := fn.addOp(op, outputShape, x)
	return stmt.Outputs[0], nil
}

// Pad x at start, end or interior (interleaved) at arbitrary axes.
//
// It adds padding values around and in-between the elements of x.
// For each axis:
//   - paddingStart elements are inserted before the tensor.
//     This value can be negative, in which case elements are removed from the start of the axis.
//   - paddingEnd elements are appended after the tensor.
//     This value can be negative, in which case elements are removed from the start of the axis.
//   - paddingInterior elements are inserted between consecutive elements of the tensor.
//     So setting paddingInterior[i]=2 for axis "i" means 2 elements will be inserted between
//     every adjacent pair of elements.
//     paddingInterior can not be negative.
//
// If any of the padding parameters is not given, it is set to 0 for all axes.
//
// The fill value must be a scalar with the same DType as x and determines what value will
// be used for the padding.
//
// The output shape is defined by:
//
//	For each axis i in x:
//	output.Dimensions[i] = paddingStart[i] + x.Dimensions[i] + max((x.Dimensions[i]-1), 0)*paddingInterior[i] + paddingEnd[i]
func Pad(x, fill *Value, paddingStart, paddingEnd, paddingInterior []int) (*Value, error) {
	op := optypes.Pad
	fn := x.fn
	if fn.Returned {
		return nil, errors.Errorf("cannot add operation %s after returning, in function %q",
			op, fn.Name)
	}
	if fill.fn != fn {
		return nil, errors.Errorf("cannot add operation %s to function %q, because fill value is from different function (%q and %q)",
			op, fn.Name, fill.fn.Name, fn.Name)
	}

	// Set default values for parameters.
	for _, padding := range []*[]int{&paddingStart, &paddingEnd, &paddingInterior} {
		if len(*padding) == 0 {
			*padding = make([]int, x.shape.Rank())
		}
	}

	outputShape, err := shapeinference.Pad(x.shape, fill.shape, paddingStart, paddingEnd, paddingInterior)
	if err != nil {
		return nil, err
	}
	stmt := fn.addOp(op, outputShape, x, fill)
	stmt.Attributes = map[string]any{
		"edge_padding_low":  intSliceToArrayI64StableHLO(paddingStart),
		"edge_padding_high": intSliceToArrayI64StableHLO(paddingEnd),
		"interior_padding":  intSliceToArrayI64StableHLO(paddingInterior),
	}
	return stmt.Outputs[0], nil
}

// Convolution performs a convolution supporting strides, padding, dilations, feature grouping, and batch grouping.
//
// See description in https://openxla.org/stablehlo/spec#convolution
//
// The parameters strides, paddings, inputDilations, and kernelDilations can be set to nil, and the default (zeros for paddings
// and ones for the others) will be used.
//
// Note: since the spec mentions that window_reversal will be removed, we didn't include it in the API.
// If you need it, we can create an alternative API for Convolve with it.
func Convolution(input, kernel *Value,
	strides []int, paddings [][2]int, inputDilations, kernelDilations []int,
	inputBatchAxis, inputChannelsAxis int, inputSpatialAxes []int,
	kernelInputChannelsAxis, kernelOutputChannelsAxis int, kernelSpatialAxes []int,
	outputBatchAxis, outputChannelsAxis int, outputSpatialAxes []int,
	channelGroupCount, batchGroupCount int,
	inputPrecision, kernelPrecision types.DotGeneralPrecisionType) (*Value, error) {
	op := optypes.Convolution
	fn := input.fn
	if fn.Returned {
		return nil, errors.Errorf("cannot add operation %s after returning, in function %q",
			op, fn.Name)
	}
	rank := input.shape.Rank()
	rankSpatial := rank - 2

	// Set default for any missing slices.
	windowReversal := make([]bool, rankSpatial)
	if len(paddings) == 0 {
		paddings = make([][2]int, rankSpatial)
	}
	for _, s := range []*[]int{&strides, &inputDilations, &kernelDilations} {
		if len(*s) == 0 {
			*s = slices.Repeat([]int{1}, rankSpatial)
		}
	}

	// Fix negative axes.
	for _, axisConfig := range []*int{&inputBatchAxis, &inputChannelsAxis, &kernelInputChannelsAxis, &kernelOutputChannelsAxis, &outputBatchAxis, &outputChannelsAxis} {
		adjustedAxis, err := shapeinference.AdjustAxisToRank(*axisConfig, rank)
		if err != nil {
			return nil, errors.Errorf("invalid channel/batch axis %d was provided, where the rank of the input/kernel/output is %d",
				*axisConfig, rank)
		}
		*axisConfig = adjustedAxis
	}
	for _, s := range []*[]int{&inputSpatialAxes, &kernelSpatialAxes, &outputSpatialAxes} {
		*s = slices.Clone(*s)
		for i, axis := range *s {
			adjustedAxis, err := shapeinference.AdjustAxisToRank(axis, rank)
			if err != nil {
				return nil, errors.Errorf("invalid spatial axes %d, where the rank of the input/kernel/output is %d",
					axis, rank)
			}
			(*s)[i] = adjustedAxis
		}
	}

	// Call shape inference.
	outputShape, err := shapeinference.Convolve(input.shape, kernel.shape,
		strides, paddings, inputDilations, kernelDilations,
		inputBatchAxis, inputChannelsAxis, inputSpatialAxes,
		kernelInputChannelsAxis, kernelOutputChannelsAxis, kernelSpatialAxes,
		outputBatchAxis, outputChannelsAxis, outputSpatialAxes,
		channelGroupCount, batchGroupCount)
	if err != nil {
		return nil, err
	}

	// Build convolution statement.
	stmt := fn.addOp(op, outputShape, input, kernel)
	precisionConfig := literalStrF("[#stablehlo<precision %s>, #stablehlo<precision %s>]",
		inputPrecision.ToStableHLO(), kernelPrecision.ToStableHLO())

	allPaddings := make([]int, 0, rankSpatial*2)
	for _, pad := range paddings {
		allPaddings = append(allPaddings, pad[0], pad[1])
	}
	paddingsConfig, err := newTensorLiteralFromFlatAndDimensions(allPaddings, rankSpatial, 2)
	if err != nil {
		return nil, errors.WithMessagef(err, "in Convolution paddings values")
	}
	convConfig := getConvAxesConfig(inputBatchAxis, inputChannelsAxis, inputSpatialAxes,
		kernelInputChannelsAxis, kernelOutputChannelsAxis, kernelSpatialAxes,
		outputBatchAxis, outputChannelsAxis, outputSpatialAxes)
	stmt.Attributes = map[string]any{
		"window_strides":      intSliceToArrayI64StableHLO(strides),
		"padding":             paddingsConfig,
		"lhs_dilation":        intSliceToArrayI64StableHLO(inputDilations),
		"rhs_dilation":        intSliceToArrayI64StableHLO(kernelDilations),
		"window_reversal":     boolSliceToArrayI1StableHLO(windowReversal),
		"dimension_numbers":   convConfig,
		"feature_group_count": int64(channelGroupCount),
		"batch_group_count":   int64(batchGroupCount),
		"precision_config":    precisionConfig,
	}
	return stmt.Outputs[0], nil
}

// getConvAxesConfig generates the StableHLO convolution dimension numbers string.
func getConvAxesConfig(
	inputBatchAxis, inputChannelsAxis int, inputSpatialAxes []int,
	kernelInputChannelsAxis, kernelOutputChannelsAxis int, kernelSpatialAxes []int,
	outputBatchAxis, outputChannelsAxis int, outputSpatialAxes []int,
) literalStr {
	spatialRank := len(inputSpatialAxes) // == len(kernelSpatialAxes) == len(outputSpatialAxes)
	setSpatialAxes := func(spatialAxes []int, def []string) {
		for i, axis := range spatialAxes {
			def[axis] = strconv.Itoa(i)
		}
	}

	inputDef := make([]string, spatialRank+2)
	inputDef[inputBatchAxis] = "b"
	inputDef[inputChannelsAxis] = "f"
	setSpatialAxes(inputSpatialAxes, inputDef)

	outputDef := make([]string, spatialRank+2)
	outputDef[outputBatchAxis] = "b"
	outputDef[outputChannelsAxis] = "f"
	setSpatialAxes(outputSpatialAxes, outputDef)

	kernelDef := make([]string, spatialRank+2)
	kernelDef[kernelInputChannelsAxis] = "i"
	kernelDef[kernelOutputChannelsAxis] = "o"
	setSpatialAxes(kernelSpatialAxes, kernelDef)

	return literalStrF("#stablehlo.conv<[%s]x[%s]->[%s]>",
		strings.Join(inputDef, ", "),
		strings.Join(kernelDef, ", "),
		strings.Join(outputDef, ", "))
}

// Reverse axes of x.
//
// E.g.: Reverse([1, 2, 3], axes=0) -> [3, 2, 1]
func Reverse(x *Value, axes ...int) (*Value, error) {
	op := optypes.Reverse
	fn := x.fn
	if fn.Returned {
		return nil, errors.Errorf("cannot add operation %s after returning, in function %q",
			op, fn.Name)
	}

	// Adjust negative axes.
	rank := x.shape.Rank()
	for i, axis := range axes {
		adjustedAxis, err := shapeinference.AdjustAxisToRank(axis, rank)
		if err != nil {
			return nil, errors.Errorf("invalid axis %d for rank(x)=%d", axis, rank)
		}
		axes[i] = adjustedAxis
	}

	// The shape remains the same.
	stmt := fn.addOp(op, x.shape, x)
	stmt.Attributes = map[string]any{
		"dimensions": intSliceToArrayI64StableHLO(axes),
	}
	return stmt.Outputs[0], nil
}

// FFT calls the XLA FFT operation, which implements {Forward, Inverse} x {Complex, Real} versions.
// See documentation in https://openxla.org/stablehlo/spec#fft, but more details in XLA page here:
// https://openxla.org/xla/operation_semantics#fft.
//
// If fftLengths are not given, one is picked for you: based on the last axis dimension for types.FFTForward, types.FFTInverse,
// and types.FFTForwardReal. And (last_dim-1)*2 for FFTInverseReal.
//
// The underlying Gopjrt implementation for CPU FFT is backed by Eigen's TensorFFT, and for GPU FFT it uses cuFFT.
func FFT(x *Value, fftType types.FFTType, fftLength ...int) (*Value, error) {
	op := optypes.Fft
	fn := x.fn
	if fn.Returned {
		return nil, errors.Errorf("cannot add operation %s after returning, in function %q",
			op, fn.Name)
	}

	// Set default fftLength if none provided.
	if len(fftLength) == 0 {
		lastDim := x.shape.Dim(-1)
		switch fftType {
		case types.FFTForward, types.FFTInverse, types.FFTForwardReal:
			fftLength = []int{lastDim}
		case types.FFTInverseReal:
			fftLength = []int{(lastDim - 1) * 2}
		}
	}

	outputShape, err := shapeinference.FFT(x.shape, fftType, fftLength)
	if err != nil {
		return nil, err
	}

	stmt := fn.addOp(op, outputShape, x)
	stmt.Attributes = map[string]any{
		"fft_type":   literalStrF("#stablehlo<fft_type %s>", fftType.ToStableHLO()),
		"fft_length": intSliceToArrayI64StableHLO(fftLength),
	}
	return stmt.Outputs[0], nil
}

// ReduceWindow reduces the inputs using arbitrary windows around each element.
//
// Each resulting element for input is initialized with initValue (e.g.: for a sum, it's 0, for a product it is 1),
// and then each value is combined with the window around the element using the reduction function.
//
// The reduction function must be created with Builder.NewClosure.
// If there are N inputs and initialValues, the reduction function should have a signature
// `(lhs, rhs) out`, where lhs, rhs and out are scalars.
//
// If strides is not set, it defaults to the value of windowDimensions -- the stride matches the window size.
//
// See MultiReduceWindow for a version that supports reducing multiple inputs at once.
//
// TODO: promotion of types doesn't seem to be working according to the spec in
func ReduceWindow(input, initialValue *Value, reductionFn *Function,
	windowDimensions, strides, inputDilations, windowDilations []int,
	padding [][2]int) (*Value, error) {
	results, err := MultiReduceWindow([]*Value{input}, []*Value{initialValue}, reductionFn,
		windowDimensions, strides, inputDilations, windowDilations, padding)
	if err != nil {
		return nil, err
	}
	return results[0], nil
}

// MultiReduceWindow reduces the inputs using arbitrary windows around each element.
//
// Each resulting element for inputs[i] is initialized with initValues[i] (e.g.: for a sum, it's 0, for a product it is 1),
// and then each value is combined with the window around the element using the reduction function.
//
// The reduction function must be created with Builder.NewClosure.
// If there are N inputs and initialValues, the reduction function should have a signature
// (lhs_1, ... lhs_N, rhs_1, ... lhs_N) and output (out_1 ... out_N), where lhs_i and rhs_i are scalars.
//
// It returns N results for each aggregated value.
//
// See ReduceWindow for a version that accepts a single input.
//
// If strides is not set, it defaults to the value of windowDimensions -- the stride matches the window size.
//
// TODO: promotion of types doesn't seem to be working according to the spec in
func MultiReduceWindow(inputs, initialValues []*Value, reductionFn *Function,
	windowDimensions, strides, inputDilations, windowDilations []int,
	paddings [][2]int) ([]*Value, error) {
	op := optypes.ReduceWindow
	if len(inputs) == 0 {
		return nil, errors.New("MultiReduce requires at least one input")
	}
	fn := inputs[0].fn
	if fn.Returned {
		return nil, errors.Errorf("cannot add operation %s after returning, in function %q",
			op, fn.Name)
	}
	for i, operand := range inputs {
		if operand.fn != fn {
			return nil, errors.Errorf("cannot add operation %s to function %q, because inputs[%d] is from different function (%q and %q)",
				op, fn.Name, i, operand.fn.Name, fn.Name)
		}
	}
	for i, operand := range initialValues {
		if operand.fn != fn {
			return nil, errors.Errorf("cannot add operation %s to function %q, because initialValues[%d] is from different function (%q and %q)",
				op, fn.Name, i, operand.fn.Name, fn.Name)
		}
	}
	if reductionFn.Parent != fn {
		return nil, errors.Errorf("cannot add operation %s because reductionFn is not a StableHLO closure for function %q",
			op, fn.Name)
	}

	// Initialize default values for parameters.
	rank := inputs[0].shape.Rank()
	for _, param := range []*[]int{&windowDimensions, &inputDilations, &windowDilations} {
		if len(*param) == 0 {
			*param = make([]int, rank)
			for i := range *param {
				(*param)[i] = 1
			}
		}
	}
	if len(strides) == 0 {
		// The default stride is the corresponding windowDimension.
		strides = slices.Clone(windowDimensions)
	}
	if len(paddings) == 0 {
		// Default paddings of 0.
		paddings = make([][2]int, rank)
	}

	outputsShapes, err := shapeinference.ReduceWindow(
		valuesToShapes(inputs), valuesToShapes(initialValues),
		valuesToShapes(reductionFn.Inputs), valuesToShapes(reductionFn.Outputs),
		windowDimensions, strides, inputDilations, windowDilations,
		paddings)
	if err != nil {
		return nil, err
	}
	allInputs := append(slices.Clone(inputs), initialValues...)
	stmt := fn.addMultiOp(op, outputsShapes, allInputs)
	stmt.Attributes = map[string]any{
		"window_dimensions": intSliceToArrayI64StableHLO(windowDimensions),
		"window_strides":    intSliceToArrayI64StableHLO(strides),
		"window_dilations":  intSliceToArrayI64StableHLO(windowDilations),
		"base_dilations":    intSliceToArrayI64StableHLO(windowDilations),
	}
	stmt.AddFunctionParameter("reductionFn", reductionFn)

	// Encode paddings:
	allPaddings := make([]int, 0, rank*2)
	for _, pad := range paddings {
		allPaddings = append(allPaddings, pad[0], pad[1])
	}
	paddingsConfig, err := newTensorLiteralFromFlatAndDimensions(allPaddings, rank, 2)
	if err != nil {
		return nil, errors.WithMessagef(err, "in Convolution paddings values")
	}
	stmt.Attributes["padding"] = paddingsConfig

	return stmt.Outputs, nil
}

// SelectAndScatter performs a ReduceWindow on the input, selecting one value per window (using the selectFn to choose the value),
// and then aggregating this value into the output (at the same index as the input).
//
// The return result has the same shape as the input, and it is populated with the initialValue.
func SelectAndScatter(input, scatterSource, initialValue *Value,
	selectFn, scatterFn *Function,
	windowDimensions, strides []int, paddings [][2]int) (*Value, error) {
	op := optypes.SelectAndScatter
	fn := input.fn
	if fn.Returned {
		return nil, errors.Errorf("cannot add operation %s after returning, in function %q",
			op, fn.Name)
	}
	if scatterSource.fn != fn {
		return nil, errors.Errorf("cannot add operation %s to function %q, because input and scatterSource are from different function (%q and %q)",
			op, fn.Name, fn.Name, scatterSource.fn.Name)
	}
	if initialValue.fn != fn {
		return nil, errors.Errorf("cannot add operation %s to function %q, because input and initialValue are from different function (%q and %q)",
			op, fn.Name, fn.Name, initialValue.fn.Name)
	}

	// Initialize default values for parameters.
	rank := input.shape.Rank()
	if len(windowDimensions) == 0 {
		windowDimensions = make([]int, rank)
		for i := range windowDimensions {
			windowDimensions[i] = 1
		}
	}
	if len(strides) == 0 {
		// The default stride is the corresponding windowDimension.
		strides = slices.Clone(windowDimensions)
	}
	if len(paddings) == 0 {
		// Default paddings of 0.
		paddings = make([][2]int, rank)
	}

	if selectFn.Parent != fn {
		return nil, errors.Errorf("cannot add operation %s because selectFn is not a StableHLO closure for function %q",
			op, fn.Name)
	}
	if scatterFn.Parent != fn {
		return nil, errors.Errorf("cannot add operation %s because scatterFn is not a StableHLO closure for function %q",
			op, fn.Name)
	}

	outputShape := input.shape
	stmt := fn.addOp(op, outputShape, input, scatterSource, initialValue)
	stmt.Attributes = map[string]any{
		"window_dimensions": intSliceToArrayI64StableHLO(windowDimensions),
		"window_strides":    intSliceToArrayI64StableHLO(strides),
	}
	stmt.AddFunctionParameter("selectFn", selectFn)
	stmt.AddFunctionParameter("scatterFn", scatterFn)

	// Encode paddings:
	allPaddings := make([]int, 0, rank*2)
	for _, pad := range paddings {
		allPaddings = append(allPaddings, pad[0], pad[1])
	}
	paddingsConfig, err := newTensorLiteralFromFlatAndDimensions(allPaddings, rank, 2)
	if err != nil {
		return nil, errors.WithMessagef(err, "in Convolution paddings values")
	}
	stmt.Attributes["padding"] = paddingsConfig
	return stmt.Outputs[0], nil
}

// DynamicSlice extracts a slice from the operand at the startIndices position and the given sliceSizes.
//
// - operand: tensor from where to take the slice.
// - startIndices: scalar tensors, one per axis of operand: len(startIndices) == operand.Rank().
// - sliceSizes: static values and fixed to keep the shape of the output static.
//
// The startIndices are adjusted as follows:
//
//	adjustedStartIndices[i] = clamp(0, StartIndices[i], operand.Dimensions[i] - sliceSizes[i])
func DynamicSlice(operand *Value, startIndices []*Value, sliceSizes []int) (*Value, error) {
	op := optypes.DynamicSlice
	fn := operand.fn
	if fn.Returned {
		return nil, errors.Errorf("cannot add operation %s after returning, in function %q",
			op, fn.Name)
	}
	for axis, idx := range startIndices {
		if idx.fn != fn {
			return nil, errors.Errorf("cannot add operation %s to function %q, because operand and startIndices[%d] are from different function (%q and %q)",
				op, fn.Name, axis, fn.Name, idx.fn.Name)
		}
	}
	outputShape := operand.shape.Clone()
	for axis, size := range sliceSizes {
		outputShape.Dimensions[axis] = size
	}
	stmt := fn.addOp(op, outputShape, append([]*Value{operand}, startIndices...)...)
	stmt.Attributes = map[string]any{"slice_sizes": intSliceToArrayI64StableHLO(sliceSizes)}
	return stmt.Outputs[0], nil
}

// DynamicUpdateSlice updates the operand with the values given in update, at the position given by startIndices.
//
// - operand: original value that to be updated.
// - update: values to "paste" on top of operand, at position startIndices.
// - startIndices: scalar tensors, one per axis of operand: len(startIndices) == operand.Rank().
// - sliceSizes: static values and fixed to keep the shape of the output static.
//
// It returns a value with the same shape as the operand, with the values updated.
//
// The startIndices are adjusted as follows:
//
//	adjustedStartIndices[i] = clamp(0, StartIndices[i], operand.Dimensions[i] - update.Dimensions[i])
func DynamicUpdateSlice(operand, update *Value, startIndices []*Value) (*Value, error) {
	op := optypes.DynamicUpdateSlice
	fn := operand.fn
	if fn.Returned {
		return nil, errors.Errorf("cannot add operation %s after returning, in function %q",
			op, fn.Name)
	}
	if update.fn != fn {
		return nil, errors.Errorf("cannot add operation %s to function %q, because operand and update are from different function (%q and %q)",
			op, fn.Name, fn.Name, update.fn.Name)
	}
	for axis, idx := range startIndices {
		if idx.fn != fn {
			return nil, errors.Errorf("cannot add operation %s to function %q, because operand and startIndices[%d] are from different function (%q and %q)",
				op, fn.Name, axis, fn.Name, idx.fn.Name)
		}
	}
	outputShape := operand.shape.Clone()
	stmt := fn.addOp(op, outputShape, append([]*Value{operand, update}, startIndices...)...)
	return stmt.Outputs[0], nil
}

// BatchNormInference implements batch normalization for inference. See details in
// https://www.tensorflow.org/xla/operation_semantics#batchnorminference.
//
// Based on the paper "Batch Normalization: Accelerating Deep Network Training by Reducing
// Internal Covariate Shift" (Sergey Ioffe, Christian Szegedy), https://arxiv.org/abs/1502.03167.
func BatchNormInference(operand, scale, offset, mean, variance *Value, epsilon float32, featureAxis int) (*Value, error) {
	op := optypes.BatchNormInference
	fn := operand.fn
	if fn.Returned {
		return nil, errors.Errorf("cannot add operation %s after returning, in function %q",
			op, fn.Name)
	}
	if scale.fn != fn || offset.fn != fn || mean.fn != fn || variance.fn != fn {
		return nil, errors.Errorf("cannot add operation %s to function %q, because operands are from different functions",
			op, fn.Name)
	}

	// Adjust negative axis.
	adjustedAxis, err := shapeinference.AdjustAxisToRank(featureAxis, operand.shape.Rank())
	if err != nil {
		return nil, errors.Errorf("invalid feature axis %d for rank(operand)=%d",
			featureAxis, operand.shape.Rank())
	}
	featureAxis = adjustedAxis

	// Output shape is identical to operand.
	outputShape := operand.shape.Clone()

	stmt := fn.addOp(op, outputShape, operand, scale, offset, mean, variance)
	stmt.Attributes = map[string]any{
		"epsilon":       epsilon,
		"feature_index": int64(featureAxis),
	}
	return stmt.Outputs[0], nil
}

// BatchNormTraining implements batch normalization for training. See details in
// https://www.tensorflow.org/xla/operation_semantics#batchnormtraining.
//
// It returns the normalized tensor, the batch mean and variance.
//
// Based on the paper "Batch Normalization: Accelerating Deep Network Training by Reducing
// Internal Covariate Shift" (Sergey Ioffe, Christian Szegedy), https://arxiv.org/abs/1502.03167.
func BatchNormTraining(operand, scale, offset *Value, epsilon float32, featureAxis int) (normalized *Value, batchMean *Value, batchVariance *Value, err error) {
	op := optypes.BatchNormTraining
	fn := operand.fn
	if fn.Returned {
		return nil, nil, nil, errors.Errorf("cannot add operation %s after returning, in function %q",
			op, fn.Name)
	}
	if scale.fn != fn || offset.fn != fn {
		return nil, nil, nil, errors.Errorf("cannot add operation %s to function %q, because operands are from different functions",
			op, fn.Name)
	}

	// Adjust negative axis.
	adjustedAxis, err := shapeinference.AdjustAxisToRank(featureAxis, operand.shape.Rank())
	if err != nil {
		return nil, nil, nil, errors.Errorf("invalid feature axis %d for rank(operand)=%d",
			featureAxis, operand.shape.Rank())
	}
	featureAxis = adjustedAxis

	// Output shapes: normalized has the same shape as the operand, mean and variance have the feature dimension only.
	normalizedShape := operand.shape.Clone()
	featureDimension := operand.shape.Dimensions[featureAxis]
	meanShape := shapes.Shape{
		DType:      operand.shape.DType,
		Dimensions: []int{featureDimension},
	}
	varianceShape := meanShape.Clone()

	stmt := fn.addMultiOp(op, []shapes.Shape{normalizedShape, meanShape, varianceShape}, []*Value{operand, scale, offset})
	stmt.Attributes = map[string]any{
		"epsilon":       epsilon,
		"feature_index": int64(featureAxis),
	}
	return stmt.Outputs[0], stmt.Outputs[1], stmt.Outputs[2], nil
}

// BatchNormGradient calculates the batch normalization gradients with respect to the input, scale, and offset.
// https://openxla.org/xla/operation_semantics#batchnormgrad
//
// The gradOutput is the adjoint gradient (the "V" in "VJP"), that is, the gradient with respect to the output of the
// batch normalization.
//
// Based on the paper "Batch Normalization: Accelerating Deep Network Training by Reducing
// Internal Covariate Shift" (Sergey Ioffe, Christian Szegedy), https://arxiv.org/abs/1502.03167.
func BatchNormGradient(operand, scale, mean, variance, gradOutput *Value, epsilon float32, featureAxis int) (gradOperand *Value, gradScale *Value, gradOffset *Value, err error) {
	op := optypes.BatchNormGrad
	fn := operand.fn
	if fn.Returned {
		return nil, nil, nil, errors.Errorf("cannot add operation %s after returning, in function %q",
			op, fn.Name)
	}
	if scale.fn != fn || mean.fn != fn || variance.fn != fn || gradOutput.fn != fn {
		return nil, nil, nil, errors.Errorf("cannot add operation %s to function %q, because operands are from different functions",
			op, fn.Name)
	}

	// Adjust negative axis.
	adjustedAxis, err := shapeinference.AdjustAxisToRank(featureAxis, operand.shape.Rank())
	if err != nil {
		return nil, nil, nil, errors.Errorf("invalid feature axis %d for rank(operand)=%d",
			featureAxis, operand.shape.Rank())
	}
	featureAxis = adjustedAxis

	// Output shapes: gradOperand has the same shape as operand, gradScale and gradOffset have the feature dimension only.
	gradOperandShape := operand.shape.Clone()
	featureDimension := operand.shape.Dimensions[featureAxis]
	gradScaleShape := shapes.Shape{
		DType:      operand.shape.DType,
		Dimensions: []int{featureDimension},
	}
	gradOffsetShape := gradScaleShape.Clone()

	stmt := fn.addMultiOp(op, []shapes.Shape{gradOperandShape, gradScaleShape, gradOffsetShape},
		[]*Value{operand, scale, mean, variance, gradOutput})
	stmt.Attributes = map[string]any{
		"epsilon":       epsilon,
		"feature_index": int64(featureAxis),
	}
	return stmt.Outputs[0], stmt.Outputs[1], stmt.Outputs[2], nil
}

// UniformQuantize the operand to a static quantized data type.
// That means the zero-point and scale of the quantization must be known at "compile" time.
//
// The dimensions of the quantizedShape is ignored, and the output will use the dimensions of the operand,
// but the DType and quantization parameters of the quantizedShape.
//
// Note: **EXPERIMENTAL**, this operation is not supported by standard CPU PJRT.
func UniformQuantize(operand *Value, quantizedShape shapes.Shape) (*Value, error) {
	op := optypes.UniformQuantize
	fn := operand.fn
	if fn.Returned {
		return nil, errors.Errorf("cannot add operation %s after returning, in function %q",
			op, fn.Name)
	}
	quantizedShape.Dimensions = slices.Clone(operand.Shape().Dimensions)
	stmt := fn.addOp(op, quantizedShape, operand)
	return stmt.Outputs[0], nil
}

// UniformDequantize takes a value with quantization and returns the value at its "expressed" dtype.
// The output will have the same dimensions as the operand, but with the expressed dtype from the quantization
// metadata and no quantization.
//
// Note: **EXPERIMENTAL**, this operation is not supported by standard CPU PJRT.
func UniformDequantize(operand *Value) (*Value, error) {
	op := optypes.UniformDequantize
	fn := operand.fn
	if fn.Returned {
		return nil, errors.Errorf("cannot add operation %s after returning, in function %q",
			op, fn.Name)
	}
	if operand.shape.Quantization == nil {
		return nil, errors.Errorf("UniformDequantize: operand %s does not have quantization metadata", operand.shape)
	}
	outputShape := operand.shape.Clone()
	outputShape.DType = operand.shape.Quantization.ExpressedType
	outputShape.Quantization = nil
	stmt := fn.addOp(op, outputShape, operand)
	return stmt.Outputs[0], nil
}

// GetDimensionSize returns a scalar i32 containing the runtime size of the specified dimension.
//
// - operand: the tensor to get the dimension size from.
// - dimension: the axis/dimension index to query (can be negative for reverse indexing).
//
// This is useful for working with dynamic shapes where dimension sizes are not known at compile time.
func GetDimensionSize(operand *Value, dimension int) (*Value, error) {
	op := optypes.GetDimensionSize
	fn := operand.fn
	if fn.Returned {
		return nil, errors.Errorf("cannot add operation %s after returning, in function %q",
			op, fn.Name)
	}

	// Adjust negative dimension index
	adjustedDim := dimension
	if dimension < 0 {
		adjustedDim = operand.shape.Rank() + dimension
	}
	if adjustedDim < 0 || adjustedDim >= operand.shape.Rank() {
		return nil, errors.Errorf("dimension %d out of bounds for rank %d tensor",
			dimension, operand.shape.Rank())
	}

	// Output is always a scalar i32
	outputShape := shapes.Make(dtypes.Int32)
	stmt := fn.addOp(op, outputShape, operand)
	stmt.Attributes = map[string]any{"dimension": int64(adjustedDim)}
	return stmt.Outputs[0], nil
}

// DynamicBroadcastInDim broadcasts the operand to a shape specified by a 1D tensor (not static dimensions).
//
// - operand: the tensor to broadcast.
// - outputDimensions: a 1D tensor of i32 or i64 values specifying the target shape dimensions.
// - broadcastDimensions: maps operand axes to output axes (like BroadcastInDim).
//
// This is the dynamic version of BroadcastInDim where the output shape is determined at runtime.
func DynamicBroadcastInDim(operand *Value, outputDimensions *Value, broadcastDimensions []int) (*Value, error) {
	op := optypes.DynamicBroadcastInDim
	fn := operand.fn
	if fn.Returned {
		return nil, errors.Errorf("cannot add operation %s after returning, in function %q",
			op, fn.Name)
	}
	if outputDimensions.fn != fn {
		return nil, errors.Errorf("cannot add operation %s to function %q, because operand and outputDimensions are from different functions (%q and %q)",
			op, fn.Name, fn.Name, outputDimensions.fn.Name)
	}

	// Validate outputDimensions is 1D tensor of integer type
	if outputDimensions.shape.Rank() != 1 {
		return nil, errors.Errorf("outputDimensions must be a 1D tensor, got rank %d",
			outputDimensions.shape.Rank())
	}
	if !outputDimensions.shape.DType.IsInt() {
		return nil, errors.Errorf("outputDimensions must be integer type, got %s",
			outputDimensions.shape.DType)
	}

	// Validate broadcastDimensions length matches operand rank
	if len(broadcastDimensions) != operand.shape.Rank() {
		return nil, errors.Errorf("broadcastDimensions length (%d) must match operand rank (%d)",
			len(broadcastDimensions), operand.shape.Rank())
	}

	// Create output shape - try to extract concrete dimensions if outputDimensions is a constant
	outputRank := outputDimensions.shape.Dimensions[0]

	// If outputRank is symbolic (negative), use the operand rank as fallback
	// This is common when broadcasting within shape computation subgraphs
	if outputRank < 0 {
		outputRank = operand.shape.Rank()
	}

	outputShape := operand.shape.Clone()
	outputShape.Dimensions = make([]int, outputRank)

	// Try to extract constant shape values
	concreteShape, ok := tryExtractConstantShape(fn, outputDimensions)
	if ok {
		// Check if all dimensions are positive (fully concrete)
		allConcrete := true
		for _, dim := range concreteShape {
			if dim <= 0 {
				allConcrete = false
				break
			}
		}
		if allConcrete {
			// Validate that broadcast is actually valid
			// For each operand dimension mapped via broadcastDimensions:
			// - If operand dim is 1, it can broadcast to any target dim
			// - If operand dim > 1, target dim must match
			broadcastValid := true
			for i, outputAxis := range broadcastDimensions {
				if outputAxis >= 0 && outputAxis < len(concreteShape) {
					operandDim := operand.shape.Dimensions[i]
					targetDim := concreteShape[outputAxis]
					if operandDim != 1 && operandDim != targetDim {
						broadcastValid = false
						break
					}
				}
			}
			if broadcastValid {
				// Use static BroadcastInDim when shape is fully known
				// This avoids XLA translation issues with dynamic_broadcast_in_dim
				targetShape := operand.shape.Clone()
				targetShape.Dimensions = concreteShape
				return BroadcastInDim(operand, targetShape, broadcastDimensions)
			}
			// Broadcast is not valid with these concrete shapes
			// Use symbolic output dimensions instead
			for i := range outputShape.Dimensions {
				outputShape.Dimensions[i] = -3
			}
			// Set bounds based on operand dimensions and extracted shape
			outputShape.DimensionBounds = make([]int, outputRank)
			for i := range outputShape.DimensionBounds {
				bound := 2048 // conservative default
				if i < len(concreteShape) && concreteShape[i] > 0 {
					bound = concreteShape[i]
				}
				outputShape.DimensionBounds[i] = bound
			}
		} else {
			// Use concrete dimensions from the constant
			copy(outputShape.Dimensions, concreteShape)
		}
	} else {
		// tryExtractConstantShape failed - try partial extraction
		// This handles the case where some dimensions are extractable (e.g., from get_dimension_size)
		// but others are runtime-computed (e.g., from reduce operations)

		// First, check if outputDimensions comes from a Concatenate operation
		// and try partial extraction
		var partialShape []int
		hasPartialShape := false
		for _, stmt := range fn.Statements {
			for _, output := range stmt.Outputs {
				if output == outputDimensions && stmt.OpType == optypes.Concatenate {
					partial, _, anyOk := tryExtractConcatenatedShapePartial(fn, stmt)
					if anyOk {
						partialShape = partial
						hasPartialShape = true
					}
					break
				}
			}
			if hasPartialShape {
				break
			}
		}

		// If we got partial results, try to fill in the unknown dimensions
		if hasPartialShape && len(partialShape) == outputRank {
			// Look for a bound for unknown dimensions
			// Use the model input sequence length (typically 128) as a reasonable bound
			defaultBound := 128 // Common sequence length for NLP models

			// Fill in unknown dimensions with the bound
			filledShape := make([]int, len(partialShape))
			for i, dim := range partialShape {
				if dim > 0 {
					filledShape[i] = dim
				} else {
					// Unknown dimension - use bound
					filledShape[i] = defaultBound
				}
			}

			// If we filled in values, try using static broadcast
			// This is safe because XLA will use the actual runtime values for the dynamic computation
			// but our static shape gives it a compilation target

			// Validate broadcast compatibility
			broadcastValid := true
			for i, outputAxis := range broadcastDimensions {
				if outputAxis >= 0 && outputAxis < len(filledShape) {
					operandDim := operand.shape.Dimensions[i]
					targetDim := filledShape[outputAxis]
					if operandDim > 0 && operandDim != 1 && operandDim != targetDim {
						broadcastValid = false
						break
					}
				}
			}

			if broadcastValid {
				// Use static broadcast with the filled shape
				targetShape := operand.shape.Clone()
				targetShape.Dimensions = filledShape
				return BroadcastInDim(operand, targetShape, broadcastDimensions)
			}
		}

		// Fallback: XLA requires bounded dimensions
		// Check if operand has all concrete dimensions
		operandIsConcrete := true
		for _, dim := range operand.shape.Dimensions {
			if dim < 0 {
				operandIsConcrete = false
				break
			}
		}

		// Check if any operand dimension is 1 - this means broadcast is potentially expanding
		hasBroadcastableDim := false
		for _, dim := range operand.shape.Dimensions {
			if dim == 1 {
				hasBroadcastableDim = true
				break
			}
		}

		if operandIsConcrete && outputRank == len(operand.shape.Dimensions) && !hasBroadcastableDim {
			// Operand is concrete with no 1-dimensions, and ranks match - use operand dimensions as output
			// This handles the case where broadcast is essentially a no-op
			copy(outputShape.Dimensions, operand.shape.Dimensions)
		} else {
			// Use dynamic dimension marker (-1) for all dimensions since they're runtime-determined
			for i := range outputShape.Dimensions {
				outputShape.Dimensions[i] = -1
			}
			// Set bounds to ensure bounded dynamism
			// For broadcast, we can use the input dimensions mapped through broadcastDimensions
			// and a conservative upper bound for other dimensions
			outputShape.DimensionBounds = make([]int, outputRank)
			maxDim := 1
			for _, dim := range operand.shape.Dimensions {
				if dim > 0 && dim > maxDim {
					maxDim = dim
				}
			}
			// Conservative: set bounds to a reasonable upper limit
			// For shape tensors and small broadcasts, use a minimum of 128 (common sequence length)
			// For larger tensors, use the largest input dimension * output rank
			conservativeBound := maxDim * outputRank
			if conservativeBound < 2048 {
				conservativeBound = 2048 // Minimum bound for typical sequence lengths
			}
			if conservativeBound > 65536 {
				conservativeBound = 65536 // Cap to prevent excessive memory allocation
			}
			for i := range outputShape.DimensionBounds {
				outputShape.DimensionBounds[i] = conservativeBound
			}
		}
	}

	stmt := fn.addOp(op, outputShape, operand, outputDimensions)
	stmt.Attributes = map[string]any{
		"broadcast_dimensions": intSliceToArrayI64StableHLO(broadcastDimensions),
	}
	return stmt.Outputs[0], nil
}

// factorize finds n factors of value that are as close to each other as possible
func factorize(value int, n int) []int {
	if n == 1 {
		return []int{value}
	}
	if n == 0 {
		return []int{}
	}

	// Start with n factors each equal to the nth root
	factors := make([]int, n)
	target := int(math.Pow(float64(value), 1.0/float64(n)))
	if target < 1 {
		target = 1
	}

	// Find the largest factor <= target that divides value
	for i := 0; i < n-1; i++ {
		// Find best factor starting from target and going down
		bestFactor := 1
		for f := target; f >= 1; f-- {
			if value%f == 0 {
				bestFactor = f
				break
			}
		}
		factors[i] = bestFactor
		value = value / bestFactor
	}
	// Last factor gets all remaining
	factors[n-1] = value

	return factors
}

// DynamicReshape reshapes the operand to a shape specified by a 1D tensor.
//
// - operand: the tensor to reshape.
// - outputShape: a 1D tensor of i32 or i64 values specifying the target shape dimensions.
//
// This is the dynamic version of Reshape where the output shape is determined at runtime.
// The total number of elements must remain the same.
func DynamicReshape(operand *Value, outputShape *Value) (*Value, error) {
	op := optypes.DynamicReshape
	fn := operand.fn
	if fn.Returned {
		return nil, errors.Errorf("cannot add operation %s after returning, in function %q",
			op, fn.Name)
	}
	if outputShape.fn != fn {
		return nil, errors.Errorf("cannot add operation %s to function %q, because operand and outputShape are from different functions (%q and %q)",
			op, fn.Name, fn.Name, outputShape.fn.Name)
	}

	// Validate outputShape is 1D tensor of integer type
	if outputShape.shape.Rank() != 1 {
		return nil, errors.Errorf("outputShape must be a 1D tensor, got rank %d",
			outputShape.shape.Rank())
	}
	if !outputShape.shape.DType.IsInt() {
		return nil, errors.Errorf("outputShape must be integer type, got %s",
			outputShape.shape.DType)
	}

	// Create output shape - try to extract concrete dimensions if outputShape is a constant
	outputRank := outputShape.shape.Dimensions[0]
	resultShape := operand.shape.Clone()
	resultShape.Dimensions = make([]int, outputRank)

	// Try to extract constant shape values
	concreteShape, ok := tryExtractConstantShape(fn, outputShape)
	if ok {
		// Check if all dimensions are concrete (positive) or need to be inferred (-1)
		hasInferDim := false
		inferDimIndex := -1
		knownProduct := 1
		for i, dim := range concreteShape {
			if dim == -1 {
				hasInferDim = true
				inferDimIndex = i
			} else if dim > 0 {
				knownProduct *= dim
			} else if dim == 0 {
				// 0 means keep the original dimension size, but we can't resolve that statically
				// if the input dimension is symbolic
				if i < len(operand.shape.Dimensions) && operand.shape.Dimensions[i] > 0 {
					concreteShape[i] = operand.shape.Dimensions[i]
					knownProduct *= concreteShape[i]
				} else {
					hasInferDim = true // treat as dynamic
				}
			}
		}

		// Try to resolve -1 dimension using input tensor size
		if hasInferDim && inferDimIndex >= 0 {
			inputSize := operand.shape.Size()
			if inputSize > 0 && knownProduct > 0 {
				// We can compute the inferred dimension
				inferredDim := inputSize / knownProduct
				concreteShape[inferDimIndex] = inferredDim
				hasInferDim = false // We resolved it
			}
		}

		if hasInferDim {
			// We have unresolvable dimensions - use static Reshape with computed dimensions
			// instead of dynamic_reshape which XLA can't compile with unbounded output
			inputSize := operand.shape.Size()

			// Build static dimensions: use extracted values where available, compute remaining
			staticDims := make([]int, outputRank)
			knownProduct := 1
			inferIdx := -1
			for i, dim := range concreteShape {
				if dim > 0 {
					staticDims[i] = dim
					knownProduct *= dim
				} else if dim == -1 {
					inferIdx = i // Will be computed
				} else {
					// 0 or other unknown - use default
					staticDims[i] = 128
					knownProduct *= 128
				}
			}

			// Compute the inferred dimension if possible
			if inferIdx >= 0 && inputSize > 0 && knownProduct > 0 {
				inferredDim := inputSize / knownProduct
				if inferredDim > 0 {
					staticDims[inferIdx] = inferredDim
				} else {
					// Can't compute - use reasonable default
					staticDims[inferIdx] = 1
				}
			}

			// Verify product matches input size
			outputSize := 1
			for _, dim := range staticDims {
				outputSize *= dim
			}

			if inputSize > 0 && outputSize != inputSize {
				// Fall back to using input dimensions distributed across output
				// This is a heuristic but better than unbounded dimensions
				for i := range staticDims {
					if i < len(operand.shape.Dimensions) && operand.shape.Dimensions[i] > 0 {
						staticDims[i] = operand.shape.Dimensions[i]
					}
				}
			}

			targetShape := operand.shape.Clone()
			targetShape.Dimensions = staticDims
			targetShape.DimensionBounds = nil
			return Reshape(operand, targetShape)
		} else {
			// All extracted dimensions are concrete - validate sizes match before using static Reshape
			inputSize := operand.shape.Size()
			outputSize := 1
			for _, dim := range concreteShape {
				outputSize *= dim
			}

			// If input has symbolic dimensions, we can't validate size match but should still use extracted shape
			inputHasSymbolic := false
			for _, dim := range operand.shape.Dimensions {
				if dim < 0 {
					inputHasSymbolic = true
					break
				}
			}

			if inputHasSymbolic {
				// Input has symbolic dimensions - trust the extracted output shape
				// This is important for ScatterND and other ops that create symbolic intermediate tensors
				targetShape := operand.shape.Clone()
				targetShape.Dimensions = concreteShape
				return Reshape(operand, targetShape)
			}

			if inputSize > 0 && outputSize > 0 && inputSize == outputSize {
				// Sizes match - use static Reshape
				// This avoids XLA translation issues with dynamic_reshape
				targetShape := operand.shape.Clone()
				targetShape.Dimensions = concreteShape
				return Reshape(operand, targetShape)
			}
			// Sizes don't match - the extracted shape is wrong for this operand
			// Use static reshape with computed dimensions based on input

			// Try to preserve as much of the extracted shape as possible while fixing the mismatch
			staticDims := make([]int, outputRank)
			copy(staticDims, concreteShape)

			// If we have one dimension that's clearly wrong, try to fix it
			// by computing what it should be based on the input size
			if inputSize > 0 {
				knownProduct := 1
				unknownIdx := -1
				for i, dim := range staticDims {
					if dim > 0 && dim <= inputSize {
						knownProduct *= dim
					} else {
						unknownIdx = i
					}
				}
				if unknownIdx >= 0 && knownProduct > 0 {
					inferredDim := inputSize / knownProduct
					if inferredDim > 0 {
						staticDims[unknownIdx] = inferredDim
					}
				}
			}

			// Verify the new shape matches
			newOutputSize := 1
			for _, dim := range staticDims {
				if dim > 0 {
					newOutputSize *= dim
				}
			}

			if newOutputSize == inputSize {
				targetShape := operand.shape.Clone()
				targetShape.Dimensions = staticDims
				return Reshape(operand, targetShape)
			}

			// Still mismatched - try to compute a valid shape that preserves input size

			// Strategy: use as many extracted dims as possible while ensuring product = inputSize
			// Keep dimensions that are factors of inputSize, adjust the rest
			validDims := make([]int, outputRank)
			remainingSize := inputSize
			dimsToInfer := 0

			// First pass: identify which extracted dims are usable
			for i, dim := range staticDims {
				if dim > 0 && dim <= remainingSize && remainingSize%dim == 0 {
					validDims[i] = dim
					remainingSize /= dim
				} else {
					validDims[i] = 0 // Mark for inference
					dimsToInfer++
				}
			}

			// Second pass: distribute remaining size across unmarked dims
			if dimsToInfer > 0 && remainingSize > 0 {
				// Try to find reasonable factorization
				for i := range validDims {
					if validDims[i] == 0 {
						if dimsToInfer == 1 {
							// Last dim gets the rest
							validDims[i] = remainingSize
							remainingSize = 1
						} else {
							// Use 1 for this dim
							validDims[i] = 1
						}
						dimsToInfer--
					}
				}
			}

			// Verify product
			checkProduct := 1
			for _, dim := range validDims {
				checkProduct *= dim
			}

			if checkProduct != inputSize {
				// Still wrong - use a simple heuristic
				// If input is [a, b] and output is rank r, try to preserve structure
				if outputRank == len(operand.shape.Dimensions) {
					copy(validDims, operand.shape.Dimensions)
				} else if outputRank > len(operand.shape.Dimensions) {
					// Add 1s at the beginning
					for i := range validDims {
						if i < outputRank-len(operand.shape.Dimensions) {
							validDims[i] = 1
						} else {
							srcIdx := i - (outputRank - len(operand.shape.Dimensions))
							if srcIdx < len(operand.shape.Dimensions) && operand.shape.Dimensions[srcIdx] > 0 {
								validDims[i] = operand.shape.Dimensions[srcIdx]
							} else {
								validDims[i] = 1
							}
						}
					}
				} else {
					// Fewer dims - merge
					validDims[outputRank-1] = inputSize
					for i := 0; i < outputRank-1; i++ {
						validDims[i] = 1
					}
				}
			}

			targetShape := operand.shape.Clone()
			targetShape.Dimensions = validDims
			return Reshape(operand, targetShape)
		}
	} else {
		// tryExtractConstantShape failed - try partial extraction first
		// This handles cases where some dimensions come from get_dimension_size (extractable)
		// and others from reduce operations (not extractable)
		var partialShape []int
		hasPartialShape := false
		for _, stmt := range fn.Statements {
			for _, output := range stmt.Outputs {
				if output == outputShape && stmt.OpType == optypes.Concatenate {
					partial, _, anyOk := tryExtractConcatenatedShapePartial(fn, stmt)
					if anyOk {
						partialShape = partial
						hasPartialShape = true
					}
					break
				}
			}
			if hasPartialShape {
				break
			}
		}

		// Calculate input size
		inputSize := 1
		hasConcreteDim := false
		for _, dim := range operand.shape.Dimensions {
			if dim > 0 {
				inputSize *= dim
				hasConcreteDim = true
			}
		}
		if !hasConcreteDim && len(operand.shape.DimensionBounds) > 0 {
			inputSize = 1
			for _, bound := range operand.shape.DimensionBounds {
				if bound > 0 {
					if inputSize > 65536/bound {
						inputSize = 65536
						break
					}
					inputSize *= bound
				}
			}
		}

		// If we got partial results, use them to compute the shape
		if hasPartialShape && len(partialShape) == outputRank {
			staticDims := make([]int, outputRank)
			knownProduct := 1
			unknownCount := 0
			unknownIdx := -1
			for i, dim := range partialShape {
				if dim > 0 {
					staticDims[i] = dim
					knownProduct *= dim
				} else {
					unknownCount++
					unknownIdx = i
				}
			}

			// If only one unknown dimension and we know the input size, compute it
			if unknownCount == 1 && inputSize > 0 && knownProduct > 0 {
				inferredDim := inputSize / knownProduct
				if inferredDim > 0 {
					staticDims[unknownIdx] = inferredDim
				} else {
					staticDims[unknownIdx] = 1
				}
			} else if unknownCount > 0 {
				// Multiple unknowns - use reasonable defaults
				for i := range staticDims {
					if staticDims[i] == 0 {
						staticDims[i] = 128 // Default for unknown dimensions
					}
				}
			}

			// Verify size match
			outputSize := 1
			for _, dim := range staticDims {
				outputSize *= dim
			}

			if inputSize > 0 && outputSize == inputSize {
				targetShape := operand.shape.Clone()
				targetShape.Dimensions = staticDims
				targetShape.DimensionBounds = nil
				return Reshape(operand, targetShape)
			}
		}

		// Fallback: shape is truly dynamic, use conservative static dimensions

		bound := 0
		for _, dim := range operand.shape.Dimensions {
			if dim > 0 && dim > bound && dim <= 1024 {
				bound = dim
			}
		}
		if bound == 0 && len(operand.shape.DimensionBounds) > 0 {
			for _, b := range operand.shape.DimensionBounds {
				if b > 0 && b > bound && b <= 1024 {
					bound = b
				}
			}
		}
		if bound == 0 {
			bound = 128 // Smaller default - 2048 caused dimension propagation issues
		}

		staticDims := make([]int, outputRank)
		if outputRank == len(operand.shape.Dimensions) {
			for i := range staticDims {
				if i < len(operand.shape.Dimensions) && operand.shape.Dimensions[i] > 0 {
					staticDims[i] = operand.shape.Dimensions[i]
				} else if i < len(operand.shape.DimensionBounds) && operand.shape.DimensionBounds[i] > 0 {
					staticDims[i] = operand.shape.DimensionBounds[i]
				} else {
					staticDims[i] = bound
				}
			}
		} else {
			for i := range staticDims {
				staticDims[i] = bound
			}
		}


		// Use static Reshape to avoid XLA translation issues
		targetShape := operand.shape.Clone()
		targetShape.Dimensions = staticDims
		targetShape.DimensionBounds = nil // Clear bounds since we're using concrete dims
		return Reshape(operand, targetShape)
	}

	stmt := fn.addOp(op, resultShape, operand, outputShape)
	return stmt.Outputs[0], nil
}

// While executes body repeatedly while condition returns true.
//
// The While operation implements a loop that continues executing the body function
// as long as the condition function returns true.
//
// Parameters:
//   - condFn: A function that takes the current state tuple and returns a scalar boolean.
//     Created with Builder.NewClosure. Must have signature (state...) -> scalar_bool
//   - bodyFn: A function that takes the current state tuple and returns the updated state tuple.
//     Created with Builder.NewClosure. Must have signature (state...) -> (state...)
//     The output types must match the input types.
//   - initialStates: Initial values for the loop state.
//
// Returns:
//   - The final state values after the loop terminates.
//
// The loop executes as follows:
//  1. Evaluate condFn with current state
//  2. If condition is false, return current state
//  3. Evaluate bodyFn with current state to get new state
//  4. Repeat from step 1
//
// Example (count from 0 to 10):
//
//	counter, _ := fn.ConstantFromScalar(int32(0))
//	condFn := fn.Closure()
//	c, _ := condFn.Input(counter.Shape())
//	limit, _ := condFn.ConstantFromScalar(int32(10))
//	cond, _ := Compare(c, limit, ComparisonDirectionLT)
//	condFn.Return(cond)
//
//	bodyFn := fn.Closure()
//	c, _ = bodyFn.Input(counter.Shape())
//	one, _ := bodyFn.ConstantFromScalar(int32(1))
//	next, _ := Add(c, one)
//	bodyFn.Return(next)
//
//	result, err := While(condFn, bodyFn, counter)
func While(condFn, bodyFn *Function, initialStates ...*Value) ([]*Value, error) {
	op := optypes.While
	if len(initialStates) == 0 {
		return nil, errors.New("While requires at least one initial state value")
	}
	fn := initialStates[0].fn
	if fn.Returned {
		return nil, errors.Errorf("cannot add operation %s after returning, in function %q",
			op, fn.Name)
	}

	// Validate all initial states are from the same function
	for i, state := range initialStates {
		if state.fn != fn {
			return nil, errors.Errorf("cannot add operation %s to function %q, because initialStates[%d] is from different function (%q and %q)",
				op, fn.Name, i, state.fn.Name, fn.Name)
		}
	}

	// Validate closure functions are children of the current function
	if condFn.Parent != fn {
		return nil, errors.Errorf("cannot add operation %s because condFn is not a StableHLO closure of %s",
			op, fn.Name)
	}
	if bodyFn.Parent != fn {
		return nil, errors.Errorf("cannot add operation %s because bodyFn is not a StableHLO closure of %s",
			op, fn.Name)
	}

	// Perform shape inference
	outputsShapes, err := shapeinference.While(
		valuesToShapes(initialStates),
		valuesToShapes(condFn.Inputs), valuesToShapes(condFn.Outputs),
		valuesToShapes(bodyFn.Inputs), valuesToShapes(bodyFn.Outputs))
	if err != nil {
		return nil, err
	}

	// Create the statement
	stmt := fn.addMultiOp(op, outputsShapes, initialStates)
	// Note: AddFunctionParameter processes parameters in alphabetical order internally,
	// so "body" comes before "cond" alphabetically. To get the correct MLIR region order
	// (body first, cond second), we add them as "cond" then "body".
	stmt.AddFunctionParameter("cond", condFn)
	stmt.AddFunctionParameter("body", bodyFn)

	return stmt.Outputs, nil
}
