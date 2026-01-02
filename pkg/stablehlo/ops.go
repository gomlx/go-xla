package stablehlo

import (
	"fmt"
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

	hasDynamic := operand.shape.IsDynamic() || shape.IsDynamic()

	// Skip size validation if any dimension is dynamic (will be validated at runtime)
	if !hasDynamic && operand.shape.Size() != shape.Size() {
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
//     Created with Function.Closure. For N inputs, must have signature
//     (lhs_0, rhs_0, lhs_1, rhs_1, ..., lhs_{N-1}, rhs_{N-1}) -> scalar_bool
//     That is, for each input tensor you get a (lhs, rhs) pair of elements being compared.
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
//	// Arguments for first input (values): lhs, rhs
//	lhsVal, _ := comparatorFn.Input(values.Shape().ScalarShape())
//	rhsVal, _ := comparatorFn.Input(values.Shape().ScalarShape())
//	// Arguments for second input (indices): lhs, rhs (not used in comparison)
//	lhsIdx, _ := comparatorFn.Input(indices.Shape().ScalarShape())
//	rhsIdx, _ := comparatorFn.Input(indices.Shape().ScalarShape())
//	_, _ = lhsIdx, rhsIdx // silence unused variable warnings
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
	copy(outputShape.Dimensions, sliceSizes)
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

// If selects between two branches based on a scalar boolean predicate.
//
// The If operation evaluates exactly one of the two branches based on the predicate value:
//   - If pred is true, the true_branch is executed
//   - If pred is false, the false_branch is executed
//
// Parameters:
//   - pred: A scalar boolean value that determines which branch to execute.
//   - trueBranch: A function to execute when pred is true.
//     Created with Function.Closure(). Must have no inputs and return one or more values.
//   - falseBranch: A function to execute when pred is false.
//     Created with Function.Closure(). Must have no inputs and return the same number
//     of values with matching shapes as trueBranch.
//
// Returns:
//   - The outputs from whichever branch was executed.
//
// Example (select max or min based on condition):
//
//	a := must(fn.ConstantFromScalar(float32(5.0)))
//	b := must(fn.ConstantFromScalar(float32(3.0)))
//	useMax := must(fn.ConstantFromScalar(true))
//
//	trueBranch := fn.Closure()
//	maxVal := must(trueBranch.ConstantFromScalar(float32(5.0))) // or compute max(a,b)
//	trueBranch.Return(maxVal)
//
//	falseBranch := fn.Closure()
//	minVal := must(falseBranch.ConstantFromScalar(float32(3.0))) // or compute min(a,b)
//	falseBranch.Return(minVal)
//
//	result, err := If(useMax, trueBranch, falseBranch)
func If(pred *Value, trueBranch, falseBranch *Function) ([]*Value, error) {
	op := optypes.If
	fn := pred.fn
	if fn.Returned {
		return nil, errors.Errorf("cannot add operation %s after returning, in function %q",
			op, fn.Name)
	}

	// Validate pred is a scalar bool
	if !pred.shape.IsScalar() || pred.shape.DType != dtypes.Bool {
		return nil, errors.Errorf("If predicate must be a scalar bool, got %s", pred.shape)
	}

	// Validate branch functions are closures of the current function
	if trueBranch.Parent != fn {
		return nil, errors.Errorf("cannot add operation %s because trueBranch is not a StableHLO closure of %s",
			op, fn.Name)
	}
	if falseBranch.Parent != fn {
		return nil, errors.Errorf("cannot add operation %s because falseBranch is not a StableHLO closure of %s",
			op, fn.Name)
	}

	// Perform shape inference
	outputsShapes, err := shapeinference.If(
		pred.shape,
		valuesToShapes(trueBranch.Inputs), valuesToShapes(trueBranch.Outputs),
		valuesToShapes(falseBranch.Inputs), valuesToShapes(falseBranch.Outputs))
	if err != nil {
		return nil, err
	}

	// Create the statement
	stmt := fn.addMultiOp(op, outputsShapes, []*Value{pred})
	// StableHLO if expects true_branch first, then false_branch
	stmt.AddFunctionParameter("true_branch", trueBranch)
	stmt.AddFunctionParameter("false_branch", falseBranch)

	return stmt.Outputs, nil
}
