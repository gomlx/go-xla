package stablehlo

import (
	"fmt"
	"io"
	"reflect"
	"slices"
	"strconv"
	"strings"

	"github.com/gomlx/go-xla/internal/optypes"
	"github.com/gomlx/go-xla/internal/shapeinference"
	"github.com/gomlx/go-xla/pkg/types/dtypes"
	"github.com/gomlx/go-xla/pkg/types/shapes"
	"github.com/gomlx/go-xla/pkg/types/shardy"
	"github.com/pkg/errors"
)

// Function represents a `func.func` in ToStableHLO.
type Function struct {
	Builder *Builder

	// Name of the function. It should not include the "@" prefix.
	Name string

	// Inputs to the function.
	Inputs []*Value

	// Outputs of the function.
	Outputs []*Value

	// Statements in the function body.
	Statements []*Statement

	// values holds all the values (e.g., %0, %1, %arg0) created in the function's scope.
	values []*Value

	// Parent of a closure function. It is only set if the function is a closure, and it's the function that created it.
	Parent *Function

	// nextArgID is the next ID to be assigned to new input arguments.
	nextArgID int

	// nextTmpID is the next ID to be assigned to new intermediary values.
	nextTmpID int

	// nextClosureID is the next ID to be assigned to new closures.
	nextClosureID int

	// Returned indicates if the function has a return statement, so it can no longer be changed.
	Returned bool
}

// findRootFn returns the root function of a function tree.
//
// There are no cases where it is more than 1-level deep, but it would work for more.
func (fn *Function) findRootFn() *Function {
	rootFn := fn
	for rootFn.Parent != nil {
		rootFn = rootFn.Parent
	}
	return rootFn
}

// newValue creates a new value with the given shape and assigns it to the next available id.
func (fn *Function) newValue(shape shapes.Shape) (v *Value) {
	rootFn := fn.findRootFn()
	v = &Value{
		fn:    fn,
		name:  strconv.Itoa(rootFn.nextTmpID),
		shape: shape,
	}
	rootFn.nextTmpID++
	fn.values = append(fn.values, v)
	return v
}

// Input creates a new input parameter for a function.
//
// If creating multiple inputs (one at a time), the order matters, since during execution of a compiled function,
// the input parameters must be given in the same order they were created.
//
// These add to the inputs already created during the function creation.
//
// It picks a default unique name for the input parameter, you can also
// provide a name with NamedInput.
func (fn *Function) Input(shape shapes.Shape) (*Value, error) {
	return fn.InputWithShardingAndAttributes(shape, nil, nil)
}

// InputWithSharding creates a new input with the given sharding specification.
func (fn *Function) InputWithSharding(shape shapes.Shape, shardingSpec *shardy.ShardingSpec) (*Value, error) {
	return fn.InputWithShardingAndAttributes(shape, shardingSpec, nil)
}

// InputWithAttributes creates a new input with the given attributes.
func (fn *Function) InputWithAttributes(shape shapes.Shape, attributes map[string]any) (*Value, error) {
	return fn.InputWithShardingAndAttributes(shape, nil, attributes)
}

// InputWithShardingAndAttributes creates a new input with the given sharding specification and attributes.
func (fn *Function) InputWithShardingAndAttributes(shape shapes.Shape, shardingSpec *shardy.ShardingSpec, attributes map[string]any) (*Value, error) {
	rootFn := fn.findRootFn()
	// Note: NamedInputWithShardingAndAttributes increments nextArgID, so we don't need to do it here.
	value, err := fn.NamedInputWithShardingAndAttributes(fmt.Sprintf("arg%d", rootFn.nextArgID), shape, shardingSpec, attributes)
	if err != nil {
		return nil, err
	}
	return value, nil
}

// NamedInput creates a new input parameter for a function with the given name -- it
// must be a unique input name.
//
// The name is passed through ConvertToValidName, which converts any non-digit or ASCII letter to an underscore.
//
// Names with the format "%d" and "arg%d" are reserved for the default input parameters.
//
// Names are used in the StableHLO code and may be helpful for debugging, but
// otherwise have no impact.
func (fn *Function) NamedInput(name string, shape shapes.Shape) (*Value, error) {
	return fn.NamedInputWithShardingAndAttributes(name, shape, nil, nil)
}

// NamedInputWithSharding creates a new input parameter for a function with the given name -- it
// must be a unique input name -- and sharding specification for distributed computation.
func (fn *Function) NamedInputWithSharding(name string, shape shapes.Shape,
	shardingSpec *shardy.ShardingSpec) (*Value, error) {
	return fn.NamedInputWithShardingAndAttributes(name, shape, shardingSpec, nil)
}

// NamedInputWithAttributes creates a new input parameter for a function with the given name and attributes.
func (fn *Function) NamedInputWithAttributes(name string, shape shapes.Shape,
	attributes map[string]any) (*Value, error) {
	return fn.NamedInputWithShardingAndAttributes(name, shape, nil, attributes)
}

// NamedInputWithShardingAndAttributes creates a new input parameter for a function with the given name -- it
// must be a unique input name -- and sharding specification for distributed computation.
//
// The shardingSpec can be nil: the default is a replicated input across all devices.
//
// The name is passed through ConvertToValidName, which converts any non-digit or ASCII letter to an underscore.
//
// Names with the format "%d" and "arg%d" are reserved for the default input parameters.
//
// Names are used in the StableHLO code and may be helpful for debugging, but otherwise have no impact.
func (fn *Function) NamedInputWithShardingAndAttributes(name string, shape shapes.Shape,
	shardingSpec *shardy.ShardingSpec, attributes map[string]any) (*Value, error) {
	value := &Value{
		fn:         fn,
		name:       ConvertToValidName(name),
		shape:      shape,
		Attributes: attributes,
	}
	for i, input := range fn.Inputs {
		if input.name == value.name {
			return nil, errors.Errorf("duplicate input name %q with input #%d", value.name, i)
		}
	}
	if shardingSpec != nil {
		if value.Attributes == nil {
			value.Attributes = make(map[string]any)
		}
		value.Attributes["sdy.sharding"] = literalStr(shardingSpec.ToValueAttribute(value.shape))
		if slices.Index(fn.Builder.meshes, shardingSpec.Mesh) == -1 {
			meshesNames := make([]string, len(fn.Builder.meshes))
			for _, mesh := range fn.Builder.meshes {
				meshesNames = append(meshesNames, mesh.Name())
			}
			return nil, errors.Errorf("sharding spec meshe %q doesn't match any of the stablehlo.Builder meshes (%s)",
				shardingSpec.Mesh, strings.Join(meshesNames, ", "))
		}
		if err := shardingSpec.ValidateShape(shape); err != nil {
			return nil, err
		}
	}
	fn.Inputs = append(fn.Inputs, value)

	// Increment nextArgID on the root function to ensure closure inputs don't conflict.
	// This is critical because closures use Input() which generates names like "arg0", "arg1", etc.
	// based on nextArgID. If we don't increment here, closures would reuse names already used
	// by the main function's named inputs (e.g., "arg0" for model inputs).
	rootFn := fn.findRootFn()
	rootFn.nextArgID++

	return value, nil
}

// ConstantFromScalar creates a new constant statement and returns the resulting value.
func (fn *Function) ConstantFromScalar(value any) (*Value, error) {
	if fn.Returned {
		return nil, errors.Errorf("Function.Return already called for %q", fn.Name)
	}

	// The shape of the constant is inferred from the value.
	dtype := dtypes.FromAny(value)
	if dtype == dtypes.INVALID {
		return nil, errors.Errorf("unsupported constant value type %T", value)
	}
	shape := shapes.Make(dtype)
	t, err := newTensorLiteralFromFlatAndDimensions(value)
	if err != nil {
		return nil, err
	}
	c := &Statement{
		Builder:  fn.Builder,
		Function: fn,
		OpType:   optypes.Constant,
		Attributes: map[string]any{
			"value": t,
		},
		Outputs: []*Value{fn.newValue(shape)},
	}
	// Set the statement reference and output index for the output value
	c.Outputs[0].stmt = c
	c.Outputs[0].outputIndex = 0
	fn.Statements = append(fn.Statements, c)
	return c.Outputs[0], nil
}

// ConstantFromFlatAndDimensions creates a new constant statement from a flat slice with the raw values and the dimensions of the shape.
func (fn *Function) ConstantFromFlatAndDimensions(flat any, dimensions ...int) (*Value, error) {
	if fn.Returned {
		return nil, errors.Errorf("Function.Return already called for %q", fn.Name)
	}
	flatV := reflect.ValueOf(flat)
	dtype := dtypes.FromGoType(flatV.Type().Elem())
	if dtype == dtypes.INVALID {
		return nil, errors.Errorf("unsupported constant flat values type %T -- expected a slice of a basic data type", flat)
	}
	shape := shapes.Make(dtype, dimensions...)
	if shape.Size() != flatV.Len() {
		return nil, errors.Errorf("flat values size %d doesn't match shape size %d (%s)", flatV.Len(), shape.Size(), shape)
	}
	c := &Statement{
		Builder:    fn.Builder,
		Function:   fn,
		OpType:     optypes.Constant,
		Attributes: make(map[string]any, 1),
		Outputs:    []*Value{fn.newValue(shape)},
	}
	// Set the statement reference and output index for the output value
	c.Outputs[0].stmt = c
	c.Outputs[0].outputIndex = 0
	var err error
	if shape.IsScalar() {
		c.Attributes["value"], err = newTensorLiteralFromFlatAndDimensions(flatV.Index(0).Interface())
	} else {
		c.Attributes["value"], err = newTensorLiteralFromFlatAndDimensions(flat, dimensions...)
	}
	if err != nil {
		return nil, err
	}
	fn.Statements = append(fn.Statements, c)
	return c.Outputs[0], nil
}

// Return adds a return statement to the function with the given return values.
// There must be at least one return value.
//
// There can be only one return statement from a Function, and it must be the last
// operation of a function.
//
// If you are doing distributed computation, you can use WithReturnShardingSpecs to specify
// the sharding requirements for each of the return values.
func (fn *Function) Return(values ...*Value) error {
	return fn.ReturnWithAttributes(values, nil)
}

// ReturnWithShardingAndAttributes is a convenience function to call ReturnWithAttributes with the given sharding
// specifications.
//
// The shardingSpecs slice of ShardingSpecs must have the same length as the values slice.
// Each ShardingSpec can be nil, in which case the default sharding is replicated across all devices.
// If shardingSpecs is nil, this behaves just like ReturnWithAttributes.
//
// The attributes slice of maps can be set to nil if there are no attributes.
func (fn *Function) ReturnWithShardingAndAttributes(values []*Value, shardingSpecs []*shardy.ShardingSpec,
	attributes []map[string]any) error {
	if len(shardingSpecs) == 0 {
		return fn.ReturnWithAttributes(values, attributes)
	}
	if len(values) != len(shardingSpecs) {
		return errors.Errorf("Function.ReturnWithShardingAndAttributes requires the same number of values and sharding specs, got %d and %d", len(values), len(shardingSpecs))
	}
	if len(attributes) == 0 {
		attributes = make([]map[string]any, len(values))
	}
	for i, shardingSpec := range shardingSpecs {
		if shardingSpec != nil {
			specLiteral := literalStr(shardingSpec.ToValueAttribute(values[i].shape))
			if attributes[i] == nil {
				attributes[i] = map[string]any{"sdy.sharding": specLiteral}
			} else {
				attributes[i]["sdy.sharding"] = specLiteral
			}
		}
	}
	return fn.ReturnWithAttributes(values, attributes)
}

// ReturnWithAttributes adds a return statement to the function with the given return values and attributes.
func (fn *Function) ReturnWithAttributes(values []*Value, attributes []map[string]any) error {
	if fn.Returned {
		return errors.Errorf("Function.Return already called for %q", fn.Name)
	}
	if len(values) == 0 {
		return errors.New("Function.Return requires at least one return value")
	}
	if len(attributes) > 0 && len(values) != len(attributes) {
		return errors.Errorf(
			"if attributes is defined (!=nil) Function.ReturnWithAttributes requires the same number of "+
				"values and attributes, got %d and %d", len(values), len(attributes))
	}
	fn.Returned = true
	outputValues := make([]*Value, len(values))
	for i, value := range values {
		if value.fn != fn {
			return errors.New("Function.Return given values that are not owned by the function")
		}
		outputValues[i] = &Value{
			fn:    fn,
			name:  value.name,
			shape: value.shape,
		}
		if len(attributes) > 0 {
			outputValues[i].Attributes = attributes[i]
		}
	}
	fn.Outputs = outputValues
	stmt := &Statement{
		Builder:  fn.Builder,
		Function: fn,
		OpType:   optypes.FuncReturn,
		Inputs:   values,
	}
	fn.Statements = append(fn.Statements, stmt)
	return nil
}

// Iota creates a constant of the given shape with increasing numbers (starting from 0)
// on the given axis. So Iota([2,2], 1) returns [[0 1][0 1]], while Iota([2,2], 0)
// returns [[0 0][1 1]].
func (fn *Function) Iota(shape shapes.Shape, axis int) (*Value, error) {
	op := optypes.Iota
	adjustedAxis, err := shapeinference.AdjustAxisToRank(axis, shape.Rank())
	if err != nil {
		return nil, errors.WithMessagef(err, "Iota axis is invalid for shape %s", shape)
	}
	stmt := fn.addOp(op, shape)
	stmt.Attributes = map[string]any{"iota_dimension": int64(adjustedAxis)}
	return stmt.Outputs[0], nil
}

// Closure creates an unnamed closure function that can be used as an argument to operations like
// Reduce, ReduceWindow, ScatterAndUpdate, etc.
//
// After created, the Closure should not be changed. But it can be used multiple times within the same parent function.
//
// The function body is defined by calling ops on the function object, as a usual Function object.
func (fn *Function) Closure() *Function {
	rootFn := fn.findRootFn()

	// the name gets overwritten in StableHLO code by the statement taking the closure as a parameter,
	// it's just for debugging purposes.
	name := fmt.Sprintf("closure%d", rootFn.nextClosureID)
	rootFn.nextClosureID++
	closureFn := fn.Builder.NewFunction(name)
	closureFn.Parent = fn
	return closureFn
}

// UseParentValue creates a reference in this closure to a value from the parent function.
// This allows closure functions (like If branches) to use values computed in the parent scope.
//
// At the StableHLO/MLIR level, closures can reference SSA (Static Single Assignment) values from their parent scope directly.
// This method enables that by creating a Value in the closure that references the same SSA name.
//
// Returns an error if:
//   - This function is not a closure (has no parent)
//   - The parentValue does not belong to the parent function
//
// Example:
//
//	// In parent function
//	x := fn.ConstantFromScalar(5.0)
//
//	// In closure (e.g., If branch)
//	closureFn := fn.Closure()
//	xInClosure := closureFn.UseParentValue(x)
//	result, _ := stablehlo.Add(xInClosure, closureFn.ConstantFromScalar(1.0))
//	closureFn.Return(result)
func (fn *Function) UseParentValue(parentValue *Value) (*Value, error) {
	if fn.Parent == nil {
		return nil, errors.New("UseParentValue can only be called on closure functions")
	}
	if parentValue == nil {
		return nil, errors.New("parentValue is nil")
	}
	// Walk up the parent chain to check if parentValue belongs to an ancestor
	currentFn := fn
	for currentFn != nil && currentFn != parentValue.fn {
		currentFn = currentFn.Parent
	}
	if currentFn == nil {
		return nil, errors.Errorf("value %q belongs to function %q, not an ancestor of function %q",
			parentValue.name, parentValue.fn.Name, fn.Name)
	}

	// Create a value in this closure that references the parent's SSA name.
	// At the MLIR level, the SSA value name is valid across nested regions.
	v := &Value{
		fn:    fn, // This value "belongs" to the closure for operation purposes
		name:  parentValue.name,
		shape: parentValue.shape,
	}
	// Note: We don't add to fn.values since this is a reference to an existing value,
	// not a new value created in this function.
	return v, nil
}

// Write the function as StableHLO code, with the given indentation.
func (fn *Function) Write(writer io.Writer, indentation string) error {
	// Create the formatting w() and we() internal functions to facilitate handling error while generating the statement code.
	var err error
	w := func(format string, args ...any) {
		// Do nothing if an error was encountered earlier.
		if err != nil {
			// No op if an error was encountered earlier
			return
		}
		_, err = fmt.Fprintf(writer, format, args...)
	}
	we := func(e elementWriter, indentation string) {
		// Do nothing if an error was encountered earlier.
		if err != nil {
			// No op if an error was encountered earlier
			return
		}
		err = e.Write(writer, indentation)
	}
	nextIndent := indentation + IndentationStep

	// Now write the function code.
	normalFunction := fn.Parent == nil
	isClosure := fn.Parent != nil
	if normalFunction {
		w("%sfunc.func @%s(", indentation, fn.Name)
	} else if isClosure {
		w("(")
	}
	for i, input := range fn.Inputs {
		if i > 0 {
			w(", ")
		}
		we(input, nextIndent)
		w(": %s", input.shape.ToStableHLO())
		writeAttributes(writer, indentation, input.Attributes, w)
	}

	if isClosure {
		w(") :\n")
	} else if normalFunction {
		w(") -> ")
		encloseOutputInParenthesis := len(fn.Outputs) > 1 || (len(fn.Outputs) == 1 && len(fn.Outputs[0].Attributes) > 0)
		if encloseOutputInParenthesis {
			w("(")
		}
		for i, output := range fn.Outputs {
			if i > 0 {
				w(", ")
			}
			w(output.shape.ToStableHLO())
			writeAttributes(writer, indentation, output.Attributes, w)
		}
		if encloseOutputInParenthesis {
			w(")")
		}
		w(" {\n")
	}

	for _, stmt := range fn.Statements {
		we(stmt, nextIndent)
		w("\n")
	}

	if normalFunction {
		w("%s}", indentation)
	}
	return err
}
