package stablehlo

import (
	"fmt"
	"strings"

	"github.com/gomlx/go-xla/internal/optypes"
	"github.com/gomlx/go-xla/pkg/types/shapes"
)

// ExtractConstantShape attempts to extract concrete integer dimensions from a shape Value.
// If the shape Value was created by a Constant operation or a Concatenate of constants,
// it extracts the integer values.
// Returns (dimensions, true) if successful, or (nil, false) if the shape is truly dynamic.
func ExtractConstantShape(fn *Function, shapeValue *Value) ([]int, bool) {
	// Search through function statements to find where shapeValue was defined
	for _, stmt := range fn.Statements {
		// Check if this statement produced the shapeValue
		for _, output := range stmt.Outputs {
			if output == shapeValue {
				return extractFromStatement(fn, stmt)
			}
		}
	}
	// Value not found in statements (shouldn't happen)
	return nil, false
}

// extractFromStatement extracts constant values from the given statement.
func extractFromStatement(fn *Function, stmt *Statement) ([]int, bool) {
	switch stmt.OpType {
	case optypes.Constant:
		// Extract the constant tensor data
		result, ok := stmt.ExtractConstantIntegers()
		if ok {
			return result, true
		}

	case optypes.Concatenate:
		// Handle Concatenate of constants (common pattern when building shape tensors)
		return extractConcatenatedShape(fn, stmt)

	case optypes.Reshape:
		// Handle Reshape - just pass through to the input
		if len(stmt.Inputs) > 0 {
			return ExtractConstantShape(fn, stmt.Inputs[0])
		}

	case optypes.GetDimensionSize:
		// Handle GetDimensionSize - extract the dimension from the input's shape
		return extractGetDimensionSize(fn, stmt)

	case optypes.Gather:
		// Handle Gather from constant tensors with constant indices
		return extractGatherConstant(fn, stmt)

	case optypes.Slice:
		// Handle Slice of constant tensors
		return extractSliceConstant(fn, stmt)

	case optypes.BroadcastInDim:
		// Handle BroadcastInDim - when broadcasting a constant to a shape
		return extractBroadcastInDimConstant(fn, stmt)

	case optypes.Multiply:
		// Handle Multiply of constants (common in shape calculations)
		return extractMultiplyConstant(fn, stmt)

	case optypes.Divide:
		// Handle Divide of constants (common in shape calculations, e.g., splitting heads)
		return extractDivideConstant(fn, stmt)

	case optypes.Select:
		// Handle Select with constant condition (common in shape calculations with -1 inference)
		return extractSelectConstant(fn, stmt)

	case optypes.Compare:
		// Handle Compare - try to evaluate constant comparisons
		return extractCompareConstant(fn, stmt)

	case optypes.Convert:
		// Handle Convert - just pass through to the input
		if len(stmt.Inputs) > 0 {
			return ExtractConstantShape(fn, stmt.Inputs[0])
		}

	case optypes.DynamicBroadcastInDim:
		// Handle DynamicBroadcastInDim - when broadcasting a constant to a dynamic shape
		return extractDynamicBroadcastConstant(fn, stmt)
	}

	// Not a constant operation
	return nil, false
}

// extractConcatenatedShape extracts constant values from a Concatenate operation
// where all inputs are constants (or nested concatenates of constants).
// This handles the common ONNX pattern where shape tensors are built by concatenating
// individual constant dimension values.
func extractConcatenatedShape(fn *Function, concatStmt *Statement) ([]int, bool) {
	// For shape tensors, we only handle axis=0 (concatenating 1D tensors or scalars)
	if dim, ok := concatStmt.ExtractIntAttribute("dimension"); ok && dim != 0 {
		return nil, false
	}

	// Extract constants from each input and concatenate them
	var result []int
	for _, input := range concatStmt.Inputs {
		dimValues, ok := ExtractConstantShape(fn, input)
		if !ok {
			// One of the inputs is not a constant
			return nil, false
		}
		result = append(result, dimValues...)
	}

	return result, true
}

// ExtractConcatenatedShapePartial is like extractConcatenatedShape but returns
// partial results. For dimensions that can't be extracted, it uses shapes.DimUnknown (-1) as a sentinel.
// Returns (result, allExtracted, anyExtracted).
func ExtractConcatenatedShapePartial(fn *Function, concatStmt *Statement) ([]int, bool, bool) {
	// For shape tensors, we only handle axis=0 (concatenating 1D tensors or scalars)
	if dim, ok := concatStmt.ExtractIntAttribute("dimension"); ok && dim != 0 {
		return nil, false, false
	}

	// Extract constants from each input and concatenate them
	var result []int
	allExtracted := true
	anyExtracted := false

	for _, input := range concatStmt.Inputs {
		dimValues, ok := ExtractConstantShape(fn, input)
		if !ok {
			// This input is not extractable - try to determine its size from shape
			// and fill with shapes.DimUnknown sentinels
			inputShape := input.Shape()
			inputSize := 1
			if inputShape.Rank() == 1 && inputShape.Dimensions[0] > 0 {
				inputSize = inputShape.Dimensions[0]
			}
			for i := 0; i < inputSize; i++ {
				result = append(result, shapes.DimUnknown)
			}
			allExtracted = false
		} else {
			result = append(result, dimValues...)
			anyExtracted = true
		}
	}

	return result, allExtracted, anyExtracted
}

// extractGatherConstant attempts to evaluate a Gather operation when both the operand
// and indices are constants. This handles the ONNX pattern where individual elements are
// extracted from a constant shape tensor using Gather operations.
func extractGatherConstant(fn *Function, gatherStmt *Statement) ([]int, bool) {
	// Gather has two inputs: operand and startIndices
	if len(gatherStmt.Inputs) != 2 {
		return nil, false
	}

	operand := gatherStmt.Inputs[0]
	startIndices := gatherStmt.Inputs[1]

	// Extract the constant operand (the tensor we're gathering from)
	operandValues, operandOk := ExtractConstantShape(fn, operand)
	// Extract the constant indices (where we're gathering from)
	indexValues, indexOk := ExtractConstantShape(fn, startIndices)

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

// extractSliceConstant attempts to evaluate a Slice operation on a constant tensor.
// This handles extracting a subrange from a constant shape tensor.
func extractSliceConstant(fn *Function, sliceStmt *Statement) ([]int, bool) {
	// Slice has one input: the operand to slice
	if len(sliceStmt.Inputs) != 1 {
		return nil, false
	}

	operand := sliceStmt.Inputs[0]

	// Extract the constant operand (the tensor we're slicing)
	operandValues, ok := ExtractConstantShape(fn, operand)
	if !ok {
		return nil, false
	}

	// Extract slice parameters from attributes
	startIndices, hasStart := sliceStmt.ExtractIntSliceAttribute("start_indices")
	limitIndices, hasLimit := sliceStmt.ExtractIntSliceAttribute("limit_indices")

	if !hasStart || !hasLimit {
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

// extractBroadcastInDimConstant attempts to extract constant values from a BroadcastInDim operation.
// This handles the case where a scalar constant (e.g., 0) is being broadcast to create a shape tensor.
// For example, ConstantOfShape creates a constant scalar and broadcasts it to the desired shape.
func extractBroadcastInDimConstant(fn *Function, broadcastStmt *Statement) ([]int, bool) {
	// BroadcastInDim has one input: the operand to broadcast
	if len(broadcastStmt.Inputs) != 1 {
		return nil, false
	}

	operand := broadcastStmt.Inputs[0]

	// Try to extract the constant value being broadcast
	operandValues, ok := ExtractConstantShape(fn, operand)
	if !ok {
		return nil, false
	}

	// If the operand is a scalar (single value), and we're broadcasting it to create a shape tensor,
	// we need to replicate that value according to the output shape
	if len(operandValues) == 1 {
		// Extract the output shape from the statement's output
		outputShape := broadcastStmt.Outputs[0].Shape()
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

// extractDynamicBroadcastConstant attempts to extract constant values from a DynamicBroadcastInDim operation.
// This handles the case where a scalar constant is being broadcast dynamically to create a shape tensor.
// For example, broadcasting constant 1 to shape [3] produces [1, 1, 1].
func extractDynamicBroadcastConstant(fn *Function, broadcastStmt *Statement) ([]int, bool) {
	// DynamicBroadcastInDim has two inputs: operand and outputDimensions
	if len(broadcastStmt.Inputs) != 2 {
		return nil, false
	}

	operand := broadcastStmt.Inputs[0]
	outputDims := broadcastStmt.Inputs[1]

	// Try to extract the constant value being broadcast
	operandValues, operandOk := ExtractConstantShape(fn, operand)
	// Try to extract the output shape from the shape tensor
	outputShape, shapeOk := ExtractConstantShape(fn, outputDims)

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
		outputShapeFromType := broadcastStmt.Outputs[0].Shape()
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

// extractGetDimensionSize attempts to extract the dimension size from a GetDimensionSize operation.
// This handles the case where the shape of an operand is known at compile time, even if the operand
// itself is not constant. For example, GetDimensionSize(x, 0) where x has shape [12, 512, 64] returns [12].
func extractGetDimensionSize(fn *Function, getDimStmt *Statement) ([]int, bool) {
	// GetDimensionSize has one input: the operand whose dimension we're extracting
	if len(getDimStmt.Inputs) != 1 {
		return nil, false
	}

	operand := getDimStmt.Inputs[0]

	// GetDimensionSize has a "dimension" attribute specifying which dimension to extract
	dimIndex, ok := getDimStmt.ExtractIntAttribute("dimension")
	if !ok {
		return nil, false
	}

	// Check that the operand has a known shape
	operandShape := operand.Shape()
	if operandShape.Rank() <= dimIndex || dimIndex < 0 {
		return nil, false
	}

	// Extract the dimension value
	dimValue := operandShape.Dimensions[dimIndex]
	if dimValue < 0 {
		// Symbolic dimension - can't materialize
		return nil, false
	}

	// Return as a single-element slice (GetDimensionSize returns a scalar, but we represent it as a 1D slice)
	return []int{dimValue}, true
}

// extractMultiplyConstant attempts to evaluate a Multiply operation when both operands
// are constants or can be extracted as constants. This handles shape calculations that
// involve multiplying dimension sizes.
func extractMultiplyConstant(fn *Function, multiplyStmt *Statement) ([]int, bool) {
	// Multiply has two inputs
	if len(multiplyStmt.Inputs) != 2 {
		return nil, false
	}

	// Try to extract both operands as constants
	lhsValues, lhsOk := ExtractConstantShape(fn, multiplyStmt.Inputs[0])
	rhsValues, rhsOk := ExtractConstantShape(fn, multiplyStmt.Inputs[1])

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

// extractSelectConstant attempts to evaluate a Select operation when the condition
// is a constant. This handles shape calculations that use conditional logic (element-wise).
func extractSelectConstant(fn *Function, selectStmt *Statement) ([]int, bool) {
	// Select has three inputs: condition, onTrue, onFalse
	if len(selectStmt.Inputs) != 3 {
		return nil, false
	}

	// Try to extract all inputs as constants
	condValues, condOk := ExtractConstantShape(fn, selectStmt.Inputs[0])
	onTrueValues, onTrueOk := ExtractConstantShape(fn, selectStmt.Inputs[1])
	onFalseValues, onFalseOk := ExtractConstantShape(fn, selectStmt.Inputs[2])

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

// extractCompareConstant attempts to evaluate a Compare operation when both operands
// are constants. Returns 1 for true, 0 for false (element-wise for tensors).
func extractCompareConstant(fn *Function, compareStmt *Statement) ([]int, bool) {
	// Compare has two inputs
	if len(compareStmt.Inputs) != 2 {
		return nil, false
	}

	// Try to extract both operands as constants
	lhsValues, lhsOk := ExtractConstantShape(fn, compareStmt.Inputs[0])
	rhsValues, rhsOk := ExtractConstantShape(fn, compareStmt.Inputs[1])

	if !lhsOk || !rhsOk {
		return nil, false
	}

	// Get the comparison direction from attributes
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

// extractDivideConstant attempts to evaluate a Divide operation when both operands
// are constants or can be extracted as constants. This handles shape calculations that
// involve dividing dimension sizes (e.g., splitting attention heads).
func extractDivideConstant(fn *Function, divideStmt *Statement) ([]int, bool) {
	// Divide has two inputs
	if len(divideStmt.Inputs) != 2 {
		return nil, false
	}

	// Try to extract both operands as constants
	lhsValues, lhsOk := ExtractConstantShape(fn, divideStmt.Inputs[0])
	rhsValues, rhsOk := ExtractConstantShape(fn, divideStmt.Inputs[1])

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
