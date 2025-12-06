package shapeinference

import (
	"fmt"
	"strings"
	"testing"

	"github.com/gomlx/go-xla/internal/optypes"
	"github.com/gomlx/go-xla/pkg/types/shapes"
	"github.com/gomlx/gopjrt/dtypes"
)

// Aliases
var (
	Bool = dtypes.Bool
	I8   = dtypes.Int8
	I32  = dtypes.Int32
	F32  = dtypes.Float32
	U64  = dtypes.Uint64

	S = shapes.Make
)

// must1 panics if there is an error.
func must1[T any](value T, err error) T {
	if err != nil {
		panic(err)
	}
	return value
}

func TestBinaryOp(t *testing.T) {
	// Invalid data types check.
	var err error
	_, err = BinaryOp(optypes.And, S(F32), S(F32))
	if err == nil {
		t.Error("expected error for And(F32, F32), got nil")
	}
	_, err = BinaryOp(optypes.Multiply, S(Bool, 1), S(Bool, 1))
	if err == nil {
		t.Error("expected error for Multiply(Bool, Bool), got nil")
	}
	_, err = BinaryOp(optypes.Xor, S(F32, 1), S(F32, 1))
	if err == nil {
		t.Error("expected error for Xor(F32, F32), got nil")
	}

	// Invalid operation type (not binary op).
	_, err = BinaryOp(optypes.Exponential, S(F32), S(F32))
	if err == nil {
		t.Error("expected error for Exponential(F32, F32), got nil")
	}

	// The same shape should be ok.
	var output shapes.Shape
	intMatrixShape := S(I8, 3, 3)
	output, err = BinaryOp(optypes.Or, intMatrixShape, intMatrixShape)
	if err != nil {
		t.Errorf("expected no error, got %v", err)
	}
	if !intMatrixShape.Equal(output) {
		t.Errorf("expected output shape %s, got %s", intMatrixShape, output)
	}

	// Scalar with matrix.
	scalarShape := S(F32)
	matrixShape := S(F32, 2, 3)
	//expectedShape := S(F32, 2, 3)
	output, err = BinaryOp(optypes.Add, scalarShape, scalarShape)
	if err != nil {
		t.Errorf("expected no error, got %v", err)
	}
	if !scalarShape.Equal(output) {
		t.Errorf("expected output shape %s, got %s", scalarShape, output)
	}
	_, err = BinaryOp(optypes.Add, scalarShape, matrixShape)
	if err == nil {
		t.Error("expected error for Add(scalar, matrix), got nil")
	}
	//require.True(t, expectedShape.Equal(output))

	// Broadcasting: not provided in StableHLO.
	shape1 := S(F32, 2, 1, 3)
	shape2 := S(F32, 1, 4, 3)
	_, err = BinaryOp(optypes.Add, shape1, shape2)
	if err == nil {
		t.Error("expected error for Add(shape1, shape2), got nil")
	}
	//expectedBroadcastShape := S(F32, 2, 4, 3)
	//require.True(t, expectedBroadcastShape.Equal(must1(BinaryOp(optypes.Multiply, shape1, shape2))))

	// Matrix with scalar.
	_, err = BinaryOp(optypes.Add, matrixShape, scalarShape)
	if err == nil {
		t.Error("expected error for Add(matrix, scalar), got nil")
	}
	//require.True(t, expectedShape.Equal(must1(BinaryOp(optypes.Add, matrixShape, scalarShape))))

	// Invalid broadcasting shapes.
	invalidShape1 := S(F32, 2, 3)
	invalidShape2 := S(F32, 3, 2)
	_, err = BinaryOp(optypes.Add, invalidShape1, invalidShape2)
	if err == nil {
		t.Error("expected error for Add(invalidShape1, invalidShape2), got nil")
	}
}

func panics(t *testing.T, f func()) {
	t.Helper()
	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic, but code did not panic")
		}
	}()
	f()
}

func TestUnaryOp(t *testing.T) {
	// Invalid data types check.
	panics(t, func() { must1(UnaryOp(optypes.Not, S(F32))) })
	panics(t, func() { must1(UnaryOp(optypes.Not, S(dtypes.Complex64))) })
	panics(t, func() { must1(UnaryOp(optypes.Negate, S(Bool))) })

	// Invalid operation type (not unary op).
	panics(t, func() { must1(UnaryOp(optypes.Add, S(F32))) })
	panics(t, func() { must1(UnaryOp(optypes.Negate, S(U64))) })

	// Valid operations
	boolShape := S(Bool, 2, 3)
	if out := must1(UnaryOp(optypes.Not, boolShape)); !boolShape.Equal(out) {
		t.Errorf("expected %s, got %s", boolShape, out)
	}

	intShape := S(I8, 3, 3)
	if out := must1(UnaryOp(optypes.Not, intShape)); !intShape.Equal(out) {
		t.Errorf("expected %s, got %s", intShape, out)
	}

	floatShape := S(F32, 2, 3)
	if out := must1(UnaryOp(optypes.Exponential, floatShape)); !floatShape.Equal(out) {
		t.Errorf("expected %s, got %s", floatShape, out)
	}
	if out := must1(UnaryOp(optypes.Negate, floatShape)); !floatShape.Equal(out) {
		t.Errorf("expected %s, got %s", floatShape, out)
	}
}

func TestGather(t *testing.T) {
	t.Run("1", func(t *testing.T) {
		operand := S(F32, 4, 3, 2, 2)
		startIndices := S(I8, 3, 3, 2)
		indexVectorAxis := 1
		offsetOutputAxes := []int{0, 3}
		collapsedSliceAxes := []int{0, 2}
		var operandBatchingAxes, startIndicesBatchingAxes []int
		startIndexMap := []int{0, 2, 3}
		sliceSizes := []int{1, 3, 1, 1}
		output, err := Gather(operand, startIndices, indexVectorAxis,
			offsetOutputAxes, collapsedSliceAxes, operandBatchingAxes,
			startIndicesBatchingAxes, startIndexMap,
			sliceSizes, false)
		if err != nil {
			t.Fatalf("expected no error, got %v", err)
		}
		fmt.Printf("\tTest 1: outputShape=%s\n", output)
		if err := output.Check(F32, 3, 3, 2, 1); err != nil {
			t.Errorf("output check failed: %v", err)
		}
	})

	t.Run("2", func(t *testing.T) {
		operand := S(F32, 3, 4, 5, 6)
		startIndices := S(U64, 7, 3, 8)
		indexVectorAxis := 1
		offsetOutputAxes := []int{1, 2}
		collapsedSliceAxes := []int{1, 2}
		operandBatchingAxes := []int{}
		startIndicesBatchingAxes := []int{}
		startIndexMap := []int{1, 2, 3}
		sliceSizes := []int{3, 1, 1, 1}
		output, err := Gather(operand, startIndices, indexVectorAxis,
			offsetOutputAxes, collapsedSliceAxes, operandBatchingAxes,
			startIndicesBatchingAxes, startIndexMap,
			sliceSizes, false)
		if err != nil {
			t.Fatalf("expected no error, got %v", err)
		}
		fmt.Printf("\tTest 2: outputShape=%s\n", output)
		if err := output.Check(F32, 7, 3, 1, 8); err != nil {
			t.Errorf("output check failed: %v", err)
		}
	})

	t.Run("3", func(t *testing.T) {
		operand := S(F32, 8, 16)
		startIndices := S(U64, 8, 1)
		indexVectorAxis := 1
		offsetOutputAxes := []int{1}
		collapsedSliceAxes := []int{0}
		operandBatchingAxes := []int{}
		startIndicesBatchingAxes := []int{}
		startIndexMap := []int{0}
		sliceSizes := []int{1, 16}
		output, err := Gather(operand, startIndices, indexVectorAxis,
			offsetOutputAxes, collapsedSliceAxes, operandBatchingAxes,
			startIndicesBatchingAxes, startIndexMap,
			sliceSizes, false)
		if err != nil {
			t.Fatalf("expected no error, got %v", err)
		}
		fmt.Printf("\tTest 3: outputShape=%s\n", output)
		if err := output.Check(F32, 8, 16); err != nil {
			t.Errorf("output check failed: %v", err)
		}
	})

	// Test from StableHLO's specification example in https://openxla.org/stablehlo/spec#gather
	t.Run("WithBatch", func(t *testing.T) {
		operand := S(F32, 2, 3, 4, 2)
		startIndices := S(dtypes.Int64, 2, 2, 3, 2)
		indexVectorAxis := 3
		offsetOutputAxes := []int{3, 4}
		collapsedSliceAxes := []int{1}
		operandBatchingAxes := []int{0}
		startIndicesBatchingAxes := []int{1}
		startIndexMap := []int{2, 1}
		sliceSizes := []int{1, 1, 2, 2}
		output, err := Gather(operand, startIndices, indexVectorAxis,
			offsetOutputAxes, collapsedSliceAxes, operandBatchingAxes,
			startIndicesBatchingAxes, startIndexMap,
			sliceSizes, false)
		if err != nil {
			t.Fatalf("expected no error, got %v", err)
		}
		fmt.Printf("\tTest 3: outputShape=%s\n", output)
		if err := output.Check(F32, 2, 2, 3, 2, 2); err != nil {
			t.Errorf("output check failed: %v", err)
		}
	})
}

func TestScatter(t *testing.T) {
	// --- Valid Cases ---

	// Case 1: Typical scatter (like TF ScatterNd)
	// Scatter 2 updates of shape [5] into operand [4, 5]
	// Indices shape [2, 1] indicates 2 indices, each pointing to 1 dimension (axis 0) of operand.
	operand1 := S(F32, 4, 5)
	indices1 := S(I8, 2, 1)  // Batch shape [2]
	updates1 := S(F32, 2, 5) // Batch shape [2]
	indexVectorAxis1 := 1
	updateWindowAxes1 := []int{1}
	insertedWindowAxes1 := []int{0}
	scatterAxesToOperandAxes1 := []int{0} // Index coordinate vector element 0 maps to operand axis 0
	expected1 := operand1
	var operandBatchingAxes, indicesBatchingAxes []int
	updateComputationInputs1 := []shapes.Shape{shapes.Make(operand1.DType), shapes.Make(operand1.DType)}
	updateComputationOutputs1 := updateComputationInputs1[:1]
	outputs1, err := Scatter([]shapes.Shape{operand1}, indices1, []shapes.Shape{updates1},
		updateWindowAxes1, insertedWindowAxes1,
		operandBatchingAxes, indicesBatchingAxes,
		scatterAxesToOperandAxes1, indexVectorAxis1,
		updateComputationInputs1, updateComputationOutputs1)
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}
	if len(outputs1) != 1 {
		t.Fatalf("expected 1 output, got %d", len(outputs1))
	}
	if !expected1.Equal(outputs1[0]) {
		t.Errorf("Valid Case 1 Failed: Expected %s, got %s", expected1, outputs1[0])
	}

	// Case 2: Scattering into a higher-rank tensor
	// Scatter updates of shape [4] into operand[i, j, :], where [i, j] comes from indices.
	// Operand: [10, 9, 8] (Rank 3)
	// Indices: [2, 3, 2] (Rank 3) -> 2x3 batch, each index is a pointer to the first 2 axes of the operand
	// Updates: [2, 3, 8] (Rank 3) -> 2x3 batch, update window shape [8]
	operand2 := S(F32, 10, 9, 8)
	indices2 := S(I32, 2, 3, 2)              // 6 indices, each is a 2D coordinate
	updates2 := S(F32, 2, 3, 8)              // 6 updates, window shape [8] matching operand's last dim
	indexVectorAxis2 := 2                    // Axis 2 of indices holds the coordinate vector [coord0, coord1]
	updateWindowAxes2 := []int{2}            // Axis 2 of updates corresponds to the window shape [8]
	insertedWindowAxes2 := []int{0, 1}       // Axis 0, 1 of operand are the dimensions determined by the indices[i,j,:]
	scatterAxesToOperandAxes2 := []int{0, 1} // index coord 0 -> operand axis 0, index coord 1 -> operand axis 1
	expected2 := operand2
	updateComputationInputs2 := []shapes.Shape{shapes.Make(operand2.DType), shapes.Make(operand2.DType)}
	updateComputationOutputs2 := updateComputationInputs2[:1]
	outputs2, err := Scatter([]shapes.Shape{operand2}, indices2, []shapes.Shape{updates2},
		updateWindowAxes2, insertedWindowAxes2,
		operandBatchingAxes, indicesBatchingAxes,
		scatterAxesToOperandAxes2, indexVectorAxis2,
		updateComputationInputs2, updateComputationOutputs2)
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}
	if len(outputs2) != 1 {
		t.Fatalf("expected 1 output, got %d", len(outputs2))
	}
	if !expected2.Equal(outputs2[0]) {
		t.Errorf("Valid Case 2 Failed: Expected %s, got %s", expected2, outputs2[0])
	}

	// Case 3: Different indexVectorAxis
	// Same as case 2, but indices are [2, 2, 3] -> indexVectorAxis is 1 and different order of axes in the operand.
	operand3 := S(F32, 10, 9, 8)
	indices3 := S(I32, 2, 2, 3) // 2x3 batch, index vector size 2, indexVecAxis=1
	updates3 := S(F32, 8, 2, 3) // Update axis [8] is "out-of-order", which should be fine.
	indexVectorAxis3 := 1       // Index vector is now axis 1
	updateWindowAxes3 := []int{0}
	insertedWindowAxes3 := []int{1, 2}
	scatterAxesToOperandAxes3 := []int{1, 2} // indices are used for different axes in the operand this time.
	expected3 := operand2                    // Still expect operand shape
	updateComputationInputs3 := []shapes.Shape{shapes.Make(operand3.DType), shapes.Make(operand3.DType)}
	updateComputationOutputs3 := updateComputationInputs3[:1]
	outputs3, err := Scatter([]shapes.Shape{operand3}, indices3, []shapes.Shape{updates3},
		updateWindowAxes3, insertedWindowAxes3,
		operandBatchingAxes, indicesBatchingAxes,
		scatterAxesToOperandAxes3, indexVectorAxis3,
		updateComputationInputs3, updateComputationOutputs3)
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}
	if len(outputs3) != 1 {
		t.Fatalf("expected 1 output, got %d", len(outputs3))
	}
	if !expected3.Equal(outputs3[0]) {
		t.Errorf("Valid Case 3 Failed (IndexVecAxis=1): Expected %s, got %s", expected3, outputs3[0])
	}

	// Case 4: No insertedWindowAxes (scattering full slices)
	// Scatter updates of shape [9] into operand [10, 9]
	operand4 := S(F32, 10, 9)
	indices4 := S(I32, 6)                 // 6 indices, coord size 1
	updates4 := S(F32, 6, 9)              // 6 updates, window shape [] (scalar)
	indexVectorAxis4 := 1                 // == indices4.Rank() -> trigger extra virtual axes to indices4.
	updateWindowAxes4 := []int{1}         // No window axes in updates (updates are scalars matching batch dims)
	insertedWindowAxes4 := []int{0}       // No window axes in operand (index selects full slice - which is scalar here)
	scatterAxesToOperandAxes4 := []int{0} // Index coord 0 -> operand axis 0
	expected4 := operand4
	updateComputationInputs4 := []shapes.Shape{shapes.Make(operand4.DType), shapes.Make(operand4.DType)}
	updateComputationOutputs4 := updateComputationInputs4[:1]
	outputs4, err := Scatter([]shapes.Shape{operand4}, indices4, []shapes.Shape{updates4},
		updateWindowAxes4, insertedWindowAxes4,
		operandBatchingAxes, indicesBatchingAxes,
		scatterAxesToOperandAxes4, indexVectorAxis4,
		updateComputationInputs4, updateComputationOutputs4)
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}
	if len(outputs4) != 1 {
		t.Fatalf("expected 1 output, got %d", len(outputs4))
	}
	if !expected4.Equal(outputs4[0]) {
		t.Errorf("Valid Case 4 Failed (No Window): Expected %s, got %s", expected4, outputs4[0])
	}

	// Case 5: rearranging the output axes:
	operand5 := S(F32, 2, 5, 2)
	indices5 := S(I32, 2, 2)
	updates5 := S(F32, 5, 2)
	indexVectorAxis5 := 1
	updateWindowAxes5 := []int{0}
	insertedWindowAxes5 := []int{0, 2}
	scatterAxesToOperandAxes5 := []int{0, 2}
	updateComputationInputs5 := []shapes.Shape{shapes.Make(operand5.DType), shapes.Make(operand5.DType)}
	updateComputationOutputs5 := updateComputationInputs5[:1]
	outputs5, err := Scatter([]shapes.Shape{operand5}, indices5, []shapes.Shape{updates5},
		updateWindowAxes5, insertedWindowAxes5,
		operandBatchingAxes, indicesBatchingAxes,
		scatterAxesToOperandAxes5, indexVectorAxis5,
		updateComputationInputs5, updateComputationOutputs5)
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}
	if len(outputs5) != 1 {
		t.Fatalf("expected 1 output, got %d", len(outputs5))
	}
	if !operand5.Equal(outputs5[0]) {
		t.Errorf("Valid Case 5 Failed (No Window): Expected %s, got %s", operand5, outputs5[0])
	}
}

func TestSlice(t *testing.T) {
	opName := "Slice"

	// --- Valid Cases ---
	// Case 1: Simple 1D slice
	operand1 := S(F32, 10)
	starts1 := []int{2}
	limits1 := []int{8}
	strides1 := []int{1}
	expected1 := S(F32, 6)
	output1, err := Slice(operand1, starts1, limits1, strides1)
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}
	if !expected1.Equal(output1) {
		t.Errorf("%s Valid Case 1 Failed: Expected %s, got %s", opName, expected1, output1)
	}

	// Case 2: 2D slice with stride 1
	operand2 := S(I32, 5, 6)
	starts2 := []int{1, 2}
	limits2 := []int{4, 5}
	strides2 := []int{1, 1}
	expected2 := S(I32, 3, 3)
	output2, err := Slice(operand2, starts2, limits2, strides2)
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}
	if !expected2.Equal(output2) {
		t.Errorf("%s Valid Case 2 Failed: Expected %s, got %s", opName, expected2, output2)
	}

	// Case 3: 3D slice with different strides
	operand3 := S(Bool, 10, 8, 6)
	starts3 := []int{1, 0, 1}
	limits3 := []int{10, 8, 6} // End index exclusive
	strides3 := []int{2, 3, 1}
	// Dim 0: (10-1)/2 = 4.5 -> 5 elements (indices 1, 3, 5, 7, 9)
	// Dim 1: (8-0)/3 = 2.66 -> 3 elements (indices 0, 3, 6)
	// Dim 2: (6-1)/1 = 5 -> 5 elements (indices 1, 2, 3, 4, 5)
	expected3 := S(Bool, 5, 3, 5)
	output3, err := Slice(operand3, starts3, limits3, strides3)
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}
	if !expected3.Equal(output3) {
		t.Errorf("%s Valid Case 3 Failed: Expected %s, got %s", opName, expected3, output3)
	}

	// Case 4: Slice resulting in size 1 dimension
	operand4 := S(F32, 10)
	starts4 := []int{5}
	limits4 := []int{6}
	strides4 := []int{1}
	expected4 := S(F32, 1)
	output4, err := Slice(operand4, starts4, limits4, strides4)
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}
	if !expected4.Equal(output4) {
		t.Errorf("%s Valid Case 4 Failed: Expected %s, got %s", opName, expected4, output4)
	}

	// Case 5: Slice taking full dimension with stride > 1
	operand5 := S(I8, 7)
	starts5 := []int{0}
	limits5 := []int{7}
	strides5 := []int{2}
	// Dim 0: (7-0)/2 = 3.5 -> 4 elements (indices 0, 2, 4, 6)
	expected5 := S(I8, 4)
	output5, err := Slice(operand5, starts5, limits5, strides5)
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}
	if !expected5.Equal(output5) {
		t.Errorf("%s Valid Case 5 Failed: Expected %s, got %s", opName, expected5, output5)
	}

	// --- Error Cases ---
	operand := S(F32, 10, 5) // Rank 2
	validStarts := []int{1, 1}
	validLimits := []int{8, 4}
	validStrides := []int{1, 1}

	// Error 1: Invalid operand DType
	_, err = Slice(shapes.Shape{DType: dtypes.InvalidDType, Dimensions: []int{10}}, []int{0}, []int{5}, []int{1})
	if err == nil {
		t.Errorf("%s Error Case 1 Failed: Invalid operand DType", opName)
	}

	// Error 2: Incorrect length for starts
	_, err = Slice(operand, []int{1}, validLimits, validStrides)
	if err == nil {
		t.Errorf("%s Error Case 2 Failed: len(starts) != rank", opName)
	}

	// Error 3: Incorrect length for limits
	_, err = Slice(operand, validStarts, []int{8}, validStrides)
	if err == nil {
		t.Errorf("%s Error Case 3 Failed: len(limits) != rank", opName)
	}

	// Error 4: Incorrect length for strides
	_, err = Slice(operand, validStarts, validLimits, []int{1})
	if err == nil {
		t.Errorf("%s Error Case 4 Failed: len(strides) != rank", opName)
	}

	// Error 5: Zero stride
	_, err = Slice(operand, validStarts, validLimits, []int{1, 0})
	if err == nil {
		t.Errorf("%s Error Case 5 Failed: Zero stride", opName)
	}

	// Error 6: Negative stride
	_, err = Slice(operand, validStarts, validLimits, []int{-1, 1})
	if err == nil {
		t.Errorf("%s Error Case 6 Failed: Negative stride", opName)
	}

	// Error 7: Start index < 0
	_, err = Slice(operand, []int{-1, 1}, validLimits, validStrides)
	if err == nil {
		t.Errorf("%s Error Case 7 Failed: Start < 0", opName)
	}

	// Error 8: Start index >= dimSize
	_, err = Slice(operand, []int{10, 1}, validLimits, validStrides)
	if err == nil {
		t.Errorf("%s Error Case 8 Failed: Start >= dimSize", opName)
	}

	// Error 9: Limit index < start index
	_, err = Slice(operand, validStarts, []int{0, 4}, validStrides) // limit[0]=0 < start[0]=1
	if err == nil {
		t.Errorf("%s Error Case 9 Failed: Limit < Start", opName)
	}

	// Error 10: Limit index > dimSize
	_, err = Slice(operand, validStarts, []int{8, 6}, validStrides) // limit[1]=6 > dimSize[1]=5
	if err == nil {
		t.Errorf("%s Error Case 10 Failed: Limit > dimSize", opName)
	}
}

func TestArgMinMax(t *testing.T) {
	// --- Valid Cases ---

	// Case 1: 1D tensor
	operand1 := S(F32, 10)
	expected1 := S(I32)
	output1 := must1(ArgMinMax(operand1, 0, I32))
	if !expected1.Equal(output1) {
		t.Errorf("Valid Case 1 Failed: Expected %s, got %s", expected1, output1)
	}

	// Case 2: 2D tensor, single axis
	operand2 := S(F32, 5, 6)
	expected2 := S(I8, 5)
	output2 := must1(ArgMinMax(operand2, 1, expected2.DType))
	if !expected2.Equal(output2) {
		t.Errorf("Valid Case 2 Failed: Expected %s, got %s", expected2, output2)
	}

	// Case 3: 3D tensor, multiple axes
	operand3 := S(F32, 4, 5, 6)
	expected3 := S(U64, 5, 6)
	output3 := must1(ArgMinMax(operand3, 0, expected3.DType))
	if !expected3.Equal(output3) {
		t.Errorf("Valid Case 3 Failed: Expected %s, got %s", expected3, output3)
	}

	// --- Error Cases ---

	// Error 1: Invalid operand DType
	panics(t, func() {
		must1(ArgMinMax(shapes.Make(dtypes.InvalidDType, 10), 0, I32))
	})

	// Error 2: Invalid axis (out of bounds)
	panics(t, func() {
		must1(ArgMinMax(operand1, 1, I32)) // operand1 is rank 1, axis 1 invalid
	})

	// Error 3: Negative axis
	panics(t, func() {
		must1(ArgMinMax(operand2, -1, I32))
	})
}

func TestIsFinite(t *testing.T) {
	// Positive case: float64 tensor.
	f64Shape := S(dtypes.Float64, 2, 3)
	output, err := IsFinite(f64Shape)
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}
	expected := S(Bool, 2, 3)
	if !expected.Equal(output) {
		t.Errorf("expected %s, got %s", expected, output)
	}

	// Check non-float type.
	_, err = IsFinite(S(Bool))
	if err == nil {
		t.Error("expected error for IsFinite(Bool), got nil")
	}
	_, err = IsFinite(S(I32))
	if err == nil {
		t.Error("expected error for IsFinite(I32), got nil")
	}
}

func TestReduceWindow(t *testing.T) {
	type testCase struct {
		name                 string
		operandShape         shapes.Shape
		windowDimensions     []int
		strides              []int
		baseDilations        []int
		windowDilations      []int
		paddings             [][2]int
		expectedShape        shapes.Shape
		expectError          bool
		errorMessageContains string // Optional: for more specific error checking
	}

	testCases := []testCase{
		{
			name:             "ScalarInput_AllNilParams_Defaults",
			operandShape:     shapes.Make(dtypes.Float32), // Rank 0
			windowDimensions: nil,                         // Should be handled as empty for rank 0
			strides:          nil,                         // Should be handled as empty for rank 0
			baseDilations:    nil,
			windowDilations:  nil,
			paddings:         nil, // Should be handled as empty for rank 0
			expectedShape:    shapes.Make(dtypes.Float32),
			expectError:      false,
		},
		{
			name:             "1D_ExplicitDefaultParams",
			operandShape:     shapes.Make(dtypes.Float32, 10),
			windowDimensions: []int{1},
			strides:          []int{1},
			baseDilations:    []int{1},
			windowDilations:  []int{1},
			paddings:         [][2]int{{0, 0}},
			// Calculation: EffIn=10, EffWin=1. PaddedEffIn=10. Num=10-1=9. Out=(9/1)+1=10.
			expectedShape: shapes.Make(dtypes.Float32, 10),
			expectError:   false,
		},
		{
			name:             "1D_NonDefault_WindowDimensions",
			operandShape:     shapes.Make(dtypes.Float32, 10),
			windowDimensions: []int{3}, // EffWin=3
			strides:          []int{3},
			baseDilations:    []int{1},
			windowDilations:  []int{1},
			paddings:         [][2]int{{0, 0}},
			// Calculation: EffIn=10, EffWin=3. PaddedEffIn=10. Num=10-3=7. Out=(7/3)+1=3.
			expectedShape: shapes.Make(dtypes.Float32, 3),
			expectError:   false,
		},
		{
			name:             "1D_NonDefault_Strides",
			operandShape:     shapes.Make(dtypes.Float32, 10),
			windowDimensions: []int{1},
			strides:          []int{2},
			baseDilations:    []int{1},
			windowDilations:  []int{1},
			paddings:         [][2]int{{0, 0}},
			// Calculation: EffIn=10, EffWin=1. PaddedEffIn=10. Num=10-1=9. Out=(9/2)+1=4+1=5.
			expectedShape: shapes.Make(dtypes.Float32, 5),
			expectError:   false,
		},
		{
			name:             "1D_NonDefault_Paddings",
			operandShape:     shapes.Make(dtypes.Float32, 10),
			windowDimensions: []int{1},
			strides:          []int{1},
			baseDilations:    []int{1},
			windowDilations:  []int{1},
			paddings:         [][2]int{{1, 1}},
			// Calculation: EffIn=10, EffWin=1. PaddedEffIn=10+1+1=12. Num=12-1=11. Out=(11/1)+1=12.
			expectedShape: shapes.Make(dtypes.Float32, 12),
			expectError:   false,
		},
		{
			name:             "1D_NonDefault_BaseDilations",
			operandShape:     shapes.Make(dtypes.Float32, 5),
			windowDimensions: []int{3}, // EffWin=3
			strides:          []int{1},
			baseDilations:    []int{2}, // EffIn=(5-1)*2+1 = 9
			windowDilations:  []int{1},
			paddings:         [][2]int{{0, 0}},
			// Calculation: PaddedEffIn=9. Num=9-3=6. Out=(6/1)+1=7.
			expectedShape: shapes.Make(dtypes.Float32, 7),
			expectError:   false,
		},
		{
			name:             "1D_NonDefault_WindowDilations",
			operandShape:     shapes.Make(dtypes.Float32, 10),
			windowDimensions: []int{3},
			strides:          []int{1},
			baseDilations:    []int{1},
			windowDilations:  []int{2}, // EffWin=(3-1)*2+1=5
			paddings:         [][2]int{{0, 0}},
			// Calculation: EffIn=10. PaddedEffIn=10. Num=10-5=5. Out=(5/1)+1=6.
			expectedShape: shapes.Make(dtypes.Float32, 6),
			expectError:   false,
		},
		{
			name:             "2D_Comprehensive_AllNonDefault",
			operandShape:     shapes.Make(dtypes.Int32, 10, 12),
			windowDimensions: []int{3, 4},
			strides:          []int{2, 3},
			baseDilations:    []int{2, 1},
			windowDilations:  []int{1, 2},
			paddings:         [][2]int{{1, 1}, {0, 2}},
			// Dim0: In=10,Win=3,Str=2,Pad=[1,1],BD=2,WD=1. EffIn=(10-1)*2+1=19. EffWin=(3-1)*1+1=3. PaddedEffIn=19+1+1=21. Num=21-3=18. Out=18/2+1=10.
			// Dim1: In=12,Win=4,Str=3,Pad=[0,2],BD=1,WD=2. EffIn=(12-1)*1+1=12. EffWin=(4-1)*2+1=7. PaddedEffIn=12+0+2=14. Num=14-7=7. Out=7/3+1=2+1=3.
			expectedShape: shapes.Make(dtypes.Int32, 10, 3),
			expectError:   false,
		},
		{
			name:             "Rank4_Image_NHWC_Style_VariedParams",
			operandShape:     shapes.Make(dtypes.Float32, 1, 20, 22, 3), // N, H, W, C
			windowDimensions: []int{1, 3, 3, 1},                         // Window on H, W
			strides:          []int{1, 2, 2, 1},                         // Stride on H, W
			baseDilations:    []int{1, 1, 1, 1},
			windowDilations:  []int{1, 1, 1, 1},
			paddings:         [][2]int{{0, 0}, {1, 0}, {0, 1}, {0, 0}}, // Padding H (low), W (high)
			// Dim0(N): In=1,Win=1,Str=1,Pad0,BD1,WD1. EffIn=1,EffWin=1.Padded=1.Num=0.Out=1.
			// Dim1(H): In=20,Win=3,Str=2,PadL=1,PadH=0,BD1,WD1. EffIn=20,EffWin=3.Padded=20+1+0=21.Num=21-3=18.Out=18/2+1=10.
			// Dim2(W): In=22,Win=3,Str=2,PadL=0,PadH=1,BD1,WD1. EffIn=22,EffWin=3.Padded=22+0+1=23.Num=23-3=20.Out=20/2+1=11.
			// Dim3(C): In=3,Win=1,Str=1,Pad0,BD1,WD1. EffIn=3,EffWin=1.Padded=3.Num=2.Out=3.
			expectedShape: shapes.Make(dtypes.Float32, 1, 10, 11, 3),
			expectError:   false,
		},
		{
			name:                 "Error_WindowTooLarge_NoPadding",
			operandShape:         shapes.Make(dtypes.Float32, 5),
			windowDimensions:     []int{6},
			strides:              []int{1},
			baseDilations:        []int{1}, // Added explicit base dilation
			windowDilations:      []int{1}, // Added explicit window dilation
			paddings:             [][2]int{{0, 0}},
			expectError:          true,
			errorMessageContains: "effective window dimension 6 for axis 0 is larger than padded effective input dimension 5",
		},
		{
			name:                 "Error_InvalidStrideZero",
			operandShape:         shapes.Make(dtypes.Float32, 5),
			windowDimensions:     []int{2},
			strides:              []int{0},
			baseDilations:        []int{1},         // Added explicit base dilation
			windowDilations:      []int{1},         // Added explicit window dilation
			paddings:             [][2]int{{0, 0}}, // Added explicit padding
			expectError:          true,
			errorMessageContains: "strides[0]=0 must be >= 1",
		},
		{
			name:                 "Error_InvalidWindowDimZero_FromNonNil",
			operandShape:         shapes.Make(dtypes.Float32, 5),
			windowDimensions:     []int{0},
			strides:              []int{1},
			baseDilations:        []int{1},         // Added explicit base dilation
			windowDilations:      []int{1},         // Added explicit window dilation
			paddings:             [][2]int{{0, 0}}, // Added explicit padding
			expectError:          true,
			errorMessageContains: "windowDimensions[0]=0 must be >= 1",
		},
		{
			name:                 "Error_NegativePadding_FromNonNil",
			operandShape:         shapes.Make(dtypes.Float32, 5),
			windowDimensions:     []int{2},
			strides:              []int{1},
			baseDilations:        []int{1}, // Added explicit base dilation
			windowDilations:      []int{1}, // Added explicit window dilation
			paddings:             [][2]int{{-1, 0}},
			expectError:          true,
			errorMessageContains: "paddings[0]=[-1, 0] must be non-negative",
		},
		{
			name:                 "Error_InvalidBaseDilationZero",
			operandShape:         shapes.Make(dtypes.Float32, 5),
			windowDimensions:     []int{2},
			strides:              []int{1},
			baseDilations:        []int{0},
			windowDilations:      []int{1},         // Added explicit window dilation
			paddings:             [][2]int{{0, 0}}, // Added explicit padding
			expectError:          true,
			errorMessageContains: "baseDilations[0]=0 must be >= 1",
		},
		{
			name:                 "Error_InvalidWindowDilationZero",
			operandShape:         shapes.Make(dtypes.Float32, 5),
			windowDimensions:     []int{2},
			strides:              []int{1},
			baseDilations:        []int{1}, // Added explicit base dilation
			windowDilations:      []int{0},
			paddings:             [][2]int{{0, 0}}, // Added explicit padding
			expectError:          true,
			errorMessageContains: "windowDilations[0]=0 must be >= 1",
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			outputShape, err := ReduceWindow(
				[]shapes.Shape{tc.operandShape},
				[]shapes.Shape{shapes.Make(tc.operandShape.DType)},
				[]shapes.Shape{shapes.Make(tc.operandShape.DType), shapes.Make(tc.operandShape.DType)},
				[]shapes.Shape{shapes.Make(tc.operandShape.DType)},
				tc.windowDimensions,
				tc.strides,
				tc.baseDilations,
				tc.windowDilations,
				tc.paddings,
			)

			if tc.expectError {
				if err == nil {
					t.Errorf("Expected an error for test case: %s", tc.name)
				}
				if tc.errorMessageContains != "" && err != nil {
					if !strings.Contains(err.Error(), tc.errorMessageContains) {
						t.Errorf("Error message mismatch for: %s, expected to contain %q, got %q", tc.name, tc.errorMessageContains, err.Error())
					}
				}
			} else {
				if err != nil {
					t.Errorf("Did not expect an error for test case: %s (error was: %v)", tc.name, err)
				}
				if len(outputShape) > 0 {
					if !tc.expectedShape.Equal(outputShape[0]) {
						t.Errorf("Mismatch in output shape for test case: %s. Expected %s, Got %s", tc.name, tc.expectedShape, outputShape)
					}
				}
			}
		})
	}
}

func TestDotGeneral(t *testing.T) {
	S := shapes.Make
	F32 := dtypes.Float32
	lhs, rhs := S(F32, 2, 3, 4, 5), S(F32, 5, 1, 2, 3)
	output, err := DotGeneral(
		lhs, []int{1}, []int{3, 0},
		rhs, []int{3}, []int{0, 2},
		F32)
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}
	// Batch dims: 5 , 2
	// Contracting dims: 3
	// Cross dims: 4 (lhs) and 1 (rhs)
	fmt.Printf("\tdotgeneral.shape=%s\n", output)
	if err := output.Check(F32, 5, 2, 4, 1); err != nil {
		t.Errorf("output check failed: %v", err)
	}
}

func TestPad(t *testing.T) {
	t.Run("Simple1D", func(t *testing.T) {
		operand := S(F32, 5)
		fillValue := S(F32) // Scalar F32
		paddingStart := []int{2}
		paddingEnd := []int{3}
		paddingInterior := []int{0}
		expected := S(F32, 10)
		output, err := Pad(operand, fillValue, paddingStart, paddingEnd, paddingInterior)
		if err != nil {
			t.Fatalf("expected no error, got %v", err)
		}
		if !expected.Equal(output) {
			t.Errorf("Expected %s, got %s", expected, output)
		}
	})

	t.Run("2DWithInterior", func(t *testing.T) {
		operand := S(F32, 3, 4)
		fillValue := S(F32)
		paddingStart := []int{1, 0}
		paddingEnd := []int{0, 2}
		paddingInterior := []int{1, 1}
		expected := S(F32, 6, 9)
		output, err := Pad(operand, fillValue, paddingStart, paddingEnd, paddingInterior)
		if err != nil {
			t.Fatalf("expected no error, got %v", err)
		}
		if !expected.Equal(output) {
			t.Errorf("Expected %s, got %s", expected, output)
		}
	})

	t.Run("3DPadding", func(t *testing.T) {
		operand := S(F32, 2, 3, 2)
		fillValue := S(F32)
		paddingStart := []int{1, 2, 0}
		paddingEnd := []int{1, 0, 1}
		paddingInterior := []int{0, 0, 0}
		expected := S(F32, 4, 5, 3)
		output, err := Pad(operand, fillValue, paddingStart, paddingEnd, paddingInterior)
		if err != nil {
			t.Fatalf("expected no error, got %v", err)
		}
		if !expected.Equal(output) {
			t.Errorf("Expected %s, got %s", expected, output)
		}
	})

	t.Run("ErrorWrongFillValueDType", func(t *testing.T) {
		operand := S(F32, 3)
		fillValue := S(I32) // Wrong dtype
		paddingStart := []int{1}
		paddingEnd := []int{1}
		paddingInterior := []int{0}
		_, err := Pad(operand, fillValue, paddingStart, paddingEnd, paddingInterior)
		if err == nil {
			t.Error("expected error for Pad with wrong fill value dtype, got nil")
		}
	})

	t.Run("ErrorNonScalarFillValue", func(t *testing.T) {
		operand := S(F32, 3)
		fillValue := S(F32, 1) // Non-scalar
		paddingStart := []int{1}
		paddingEnd := []int{1}
		paddingInterior := []int{0}
		_, err := Pad(operand, fillValue, paddingStart, paddingEnd, paddingInterior)
		if err == nil {
			t.Error("expected error for Pad with non-scalar fill value, got nil")
		}
	})

	t.Run("ErrorMismatchedRank", func(t *testing.T) {
		operand := S(F32, 3)
		fillValue := S(F32)
		paddingStart := []int{1, 0}
		paddingEnd := []int{1, 1}
		paddingInterior := []int{0, 0}
		_, err := Pad(operand, fillValue, paddingStart, paddingEnd, paddingInterior)
		if err == nil {
			t.Error("expected error for Pad with mismatched rank, got nil")
		}
	})

	t.Run("NegativePadding", func(t *testing.T) {
		operand := S(F32, 3)
		fillValue := S(F32)
		paddingStart := []int{-1}
		paddingEnd := []int{-1}
		paddingInterior := []int{0}
		expected := S(F32, 1)
		output, err := Pad(operand, fillValue, paddingStart, paddingEnd, paddingInterior)
		if err != nil {
			t.Fatalf("expected no error, got %v", err)
		}
		if !expected.Equal(output) {
			t.Errorf("Expected %s, got %s", expected, output)
		}
	})

	t.Run("ErrorNegativeInterior", func(t *testing.T) {
		operand := S(F32, 3)
		fillValue := S(F32)
		paddingStart := []int{0}
		paddingEnd := []int{0}
		paddingInterior := []int{-1}
		_, err := Pad(operand, fillValue, paddingStart, paddingEnd, paddingInterior)
		if err == nil {
			t.Error("expected error for Pad with negative interior, got nil")
		}
	})
}

func TestCollectiveOps(t *testing.T) {
	operand := S(F32, 2, 4)
	replicaGroups := [][]int{{0, 1}, {2, 3}}

	t.Run("AllGather", func(t *testing.T) {
		output, err := AllGather(operand, replicaGroups, 1)
		if err != nil {
			t.Fatalf("expected no error, got %v", err)
		}
		expected := S(F32, 2, 8)
		if !expected.Equal(output) {
			t.Errorf("Expected %s, got %s", expected, output)
		}

		_, err = AllGather(operand, replicaGroups, 2)
		if err == nil {
			t.Error("expected error for AllGather with invalid dimension, got nil")
		}
	})

	t.Run("AllToAll", func(t *testing.T) {
		output, err := AllToAll(operand, replicaGroups, 1, 0, 2)
		if err != nil {
			t.Fatalf("expected no error, got %v", err)
		}
		expected := S(F32, 4, 2)
		if !expected.Equal(output) {
			t.Errorf("Expected %s, got %s", expected, output)
		}

		_, err = AllToAll(operand, replicaGroups, 2, 0, 2)
		if err == nil {
			t.Error("expected error for AllToAll with invalid dimension, got nil")
		}
	})

	t.Run("CollectivePermute", func(t *testing.T) {
		output, err := CollectivePermute(operand, [][2]int{{0, 1}})
		if err != nil {
			t.Fatalf("expected no error, got %v", err)
		}
		if !operand.Equal(output) {
			t.Errorf("Expected %s, got %s", operand, output)
		}
	})
}
