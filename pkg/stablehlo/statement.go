package stablehlo

import (
	"fmt"
	"io"
	"maps"
	"math"
	"reflect"
	"slices"
	"strconv"
	"strings"

	"github.com/gomlx/go-xla/internal/optypes"
	"github.com/gomlx/go-xla/pkg/types/dtypes"
	"github.com/gomlx/go-xla/pkg/types/dtypes/bfloat16"
	"github.com/gomlx/go-xla/pkg/types/shapes"
	"github.com/pkg/errors"
	"github.com/x448/float16"
)

// Statement represents a single operation line in ToStableHLO.
type Statement struct {
	Builder  *Builder
	Function *Function

	// OpType is the type of the operation.
	OpType optypes.OpType

	// Inputs to the operation.
	Inputs []*Value

	// Attributes of the operation.
	Attributes map[string]any

	// FunctionParameters for statements with operations like Reduce, ReduceWindow, ScatterAndUpdate, etc.
	FunctionParameters      []*Function
	FunctionParametersNames []string

	// Outputs of the operation. It may be nil for operations like func.return.
	Outputs []*Value
}

func (s *Statement) AddFunctionParameter(name string, inlineFn *Function) {
	s.FunctionParameters = append(s.FunctionParameters, inlineFn)
	s.FunctionParametersNames = append(s.FunctionParametersNames, name)
}

// Write writes a string representation of the statement to the given writer.
func (s *Statement) Write(writer io.Writer, indentation string) error {
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
	nextIndentation := indentation + IndentationStep

	// Output values are written first:
	w("%s", indentation) // IndentationStep of functions.
	if len(s.Outputs) > 0 {
		for i, output := range s.Outputs {
			if i > 0 {
				w(", ")
			}
			we(output, nextIndentation)
		}
		w(" = ")
	}

	// Write op name and arguments:
	w("%q(", s.OpType.ToStableHLO())
	for i, input := range s.Inputs {
		if i > 0 {
			w(", ")
		}
		we(input, nextIndentation)
	}
	w(")")

	// Write function parameters:
	if len(s.FunctionParameters) > 0 {
		w(" ({\n%s", nextIndentation)
		for i, param := range s.FunctionParameters {
			if i > 0 {
				w("%s}, {\n%s", indentation, nextIndentation)
			}
			w("^%s", s.FunctionParametersNames[i])
			we(param, nextIndentation+IndentationStep)
		}
		w("%s})", indentation)
	}

	// Write attributes:
	writeAttributes(writer, indentation, s.Attributes, w)

	// Write signature:
	w(" : (")
	for i, input := range s.Inputs {
		if i > 0 {
			w(", ")
		}
		w(input.shape.ToStableHLO())
	}
	w(")")
	w(" -> ")
	if len(s.Outputs) == 0 {
		w("()")
	} else {
		// There are outputs: we use "(" and ")" only if there are more than one.
		if len(s.Outputs) > 1 {
			w("(")
		}
		for i, output := range s.Outputs {
			if i > 0 {
				w(", ")
			}
			w(output.shape.ToStableHLO())
		}
		if len(s.Outputs) > 1 {
			w(")")
		}
	}

	return err
}

// writeAttributes writes a map of attributes to the writer.
// The w function is the one provided by the caller to handle errors.
func writeAttributes(writer io.Writer, indentation string, attributes map[string]any, w func(format string, args ...any)) {
	if len(attributes) == 0 {
		return
	}
	nextIndentation := indentation + IndentationStep
	if len(attributes) == 1 {
		for key, value := range attributes {
			literalValue := literalToStableHLO(value)
			if strings.Index(literalValue, "\n") == -1 {
				w(" { %s = %s }", key, literalValue)
			} else {
				literalValue = strings.ReplaceAll(literalValue, "\n", "\n"+nextIndentation)
				w(" {\n%s%s = %s\n  }", nextIndentation, key, literalValue)
			}
		}
	} else {
		// One attribute per line:
		w(" {")
		keys := slices.Collect(maps.Keys(attributes))
		slices.Sort(keys)
		for i, key := range keys {
			if i > 0 {
				w(",")
			}
			w("\n%s%s = %s", nextIndentation, key, literalToStableHLO(attributes[key]))
		}
		w("\n%s}", indentation)
	}
}

// hasToStableHLO is implemented by types that can be converted to a stablehlo string.
type hasToStableHLO interface {
	ToStableHLO() string
}

// literalStr represents a value already rendered in StableHLO format.
type literalStr string

// literalStrF format the string into a literalStr.
// It also replaces tabs by IndentionStep.
func literalStrF(format string, args ...any) literalStr {
	str := fmt.Sprintf(format, args...)
	str = strings.ReplaceAll(str, "\t", IndentationStep)
	return literalStr(str)
}

// ToStableHLO returns the string representation of the literal.
func (str literalStr) ToStableHLO() string {
	return string(str)
}

// literalToStableHLO converts a literal value, usually used in attributes, to its ToStableHLO string representation.
func literalToStableHLO(attr any) string {
	switch v := attr.(type) {
	case string:
		return fmt.Sprintf("%q", v)
	case float32, float64, int, int8, int16, int32, int64, uint8, uint16, uint32, uint64:
		dtype := dtypes.FromAny(v)
		return fmt.Sprintf("%s : %s",
			podToStableHLO(v),
			dtype.ToStableHLO())

	case bool:
		return fmt.Sprintf("%s", podToStableHLO(v))

	case hasToStableHLO:
		// For types that implement their own conversion to stablehlo, use that.
		return v.ToStableHLO()

	default:
		return fmt.Sprintf("Unknown literal type: %t %#v", v, v)
	}
}

// intSliceToStableHLO converts a slice of ints to a string with comma-separated values, as used
// by StableHLO for attribute values that are an array of ints.
func intSliceToStableHLO(ints []int) literalStr {
	str := fmt.Sprint(ints) // Produces "[1 2 3]"
	return literalStr(strings.Replace(str, " ", ", ", -1))
}

// intSliceToArrayI64StableHLO converts a slice of ints to a string with comma-separated values, as used
// by StableHLO for attribute values that are an array of int64.
func intSliceToArrayI64StableHLO(ints []int) literalStr {
	var sb strings.Builder
	sb.WriteString("array<i64")
	for i, v := range ints {
		if i == 0 {
			sb.WriteString(": ")
		} else {
			sb.WriteString(", ")
		}
		sb.WriteString(strconv.Itoa(v))
	}
	sb.WriteString(">")
	return literalStr(sb.String())
}

// boolSliceToArrayI1StableHLO converts a slice of bool to a string with comma-separated values, as used
// by StableHLO for attribute values that are an array of int64.
func boolSliceToArrayI1StableHLO(values []bool) literalStr {
	var sb strings.Builder
	sb.WriteString("array<i1")
	for i, v := range values {
		if i == 0 {
			sb.WriteString(": ")
		} else {
			sb.WriteString(", ")
		}
		sb.WriteString(fmt.Sprintf("%v", v))
	}
	sb.WriteString(">")
	return literalStr(sb.String())
}

func float32IsFinite(f float32) bool {
	return !math.IsInf(float64(f), 0) && !math.IsNaN(float64(f))
}

func float32AsHex(f float32) string {
	return fmt.Sprintf("%#x", math.Float32bits(f))
}

func float64IsFinite(f float64) bool {
	return !math.IsInf(f, 0) && !math.IsNaN(f)
}

func float64AsHex(f float64) string {
	return fmt.Sprintf("%#x", math.Float64bits(f))
}

// floatToStableHLO converts a float to a string. f must be a float32 or float64.
func floatToStableHLO(fAny any) string {
	var f64 float64

	// Each 16-bit float will have it's own infinity/NaN representation, but otherwise they can all be converted
	// to float64 before printing.
	if f16, ok := fAny.(float16.Float16); ok {
		f32 := f16.Float32()
		if !float32IsFinite(f32) {
			return fmt.Sprintf("%#x", uint16(f16))
		}
		f64 = float64(f32)
	} else if bf16, ok := fAny.(bfloat16.BFloat16); ok {
		f32 := bf16.Float32()
		if !float32IsFinite(f32) {
			return fmt.Sprintf("%#x", uint16(bf16))
		}
		f64 = float64(f32)
	} else if f32, ok := fAny.(float32); ok {
		if !float32IsFinite(f32) {
			return float32AsHex(f32)
		}
		f64 = float64(f32)
	} else {
		f64 = fAny.(float64)
		if !float64IsFinite(fAny.(float64)) {
			return float64AsHex(fAny.(float64))
		}
	}

	if math.IsNaN(f64) {
		return "nan"
	}
	if math.IsInf(f64, 0) {
		return "+inf"
	}
	if math.IsInf(f64, 1) {
		return "-inf"
	}

	// StableHLO requires a decimal point, but Go is not able to format like that (%f also doesn't work for exponents
	// and arbitrarily long decimals), so it requires some editing.
	s := fmt.Sprintf("%g", f64)
	// - A valid float literal must contain a decimal point '.' or an exponent 'e'/'E'.
	//   If it has neither, it's an integer like "42", so we add ".0".
	if !strings.ContainsAny(s, ".eE") {
		return s + ".0"
	}

	// - If it has an exponent but no decimal (the problematic case, e.g., "1e-06"),
	//   we need to insert one.
	if strings.ContainsAny(s, "eE") && !strings.Contains(s, ".") {
		eIndex := strings.IndexAny(s, "eE")
		return s[:eIndex] + ".0" + s[eIndex:]
	}
	return s
}

// podToStableHLO convert a POD (plain-old-data) value (scalar floats, ints, bool and complex) to a stableHLO string,
// with no types attached.
func podToStableHLO(pod any) string {
	switch v := pod.(type) {
	case float16.Float16, bfloat16.BFloat16, float32, float64:
		return floatToStableHLO(v)

	case int, int8, int16, int32, int64, uint8, uint16, uint32, uint64:
		return fmt.Sprintf("%d", v)

	case bool:
		if v {
			return "true"
		}
		return "false"

	case complex64, complex128:
		var c complex128
		if c64, ok := v.(complex64); ok {
			c = complex128(c64)
		} else {
			c = v.(complex128)
		}
		return fmt.Sprintf("(%s, %s)",
			floatToStableHLO(real(c)), floatToStableHLO(imag(c)))

	default:
		return fmt.Sprintf("*don't know how to present data type*: %t %#v", v, v)
	}
}

// tensorLiteral represents a literal tensor value, used to define constants.
//
// It has a different representation than other literals.
type tensorLiteral struct {
	// value is either a scalar value or a flat slice of the values.
	value any

	// dims has the dimensions of the tensor or nil if the value is a scalar.
	dims []int
}

// newTensorLiteralFromFlatAndDimensions creates a new tensorLiteral that can be used to render constants.
//
// Args:
// - value is either a scalar value or a flat slice of the values.
// - dims has the dimensions of the tensor or nil if the value is a scalar.
func newTensorLiteralFromFlatAndDimensions(value any, dims ...int) (t tensorLiteral, err error) {
	size := 1
	for _, dim := range dims {
		size *= dim
	}

	valueV := reflect.ValueOf(value)
	if valueV.Kind() != reflect.Slice && valueV.Kind() != reflect.Array {
		if len(dims) != 0 {
			err = errors.Errorf("flat value is not a slice or array for a non-scalar shape, got %T instead (%s)", value, valueV.Kind())
			return
		}
		// Simple scalar value:
		return tensorLiteral{value: value}, nil
	}

	if valueV.Len() != size {
		err = errors.Errorf("expected %d flat elements for shape %v, got %d instead", size, dims, valueV.Len())
		return
	}
	return tensorLiteral{value: value, dims: dims}, nil
}

// ToStableHLO returns the string representation of the tensor literal.
func (t tensorLiteral) ToStableHLO() string {
	valueV := reflect.ValueOf(t.value)
	var shape shapes.Shape
	if valueV.Kind() != reflect.Slice && valueV.Kind() != reflect.Array {
		// Scalar value:
		shape.DType = dtypes.FromGoType(valueV.Type())
		return fmt.Sprintf("dense<%s> : %s", podToStableHLO(t.value), shape.ToStableHLO())
	}

	shape.DType = dtypes.FromGoType(valueV.Type().Elem())
	shape.Dimensions = slices.Clone(t.dims)
	var flatIdx int
	var sb strings.Builder
	recursiveTensorToStableHLO(valueV, shape, flatIdx, 0, &sb)
	return fmt.Sprintf("dense<%s> : %s", sb.String(), shape.ToStableHLO())
}

func recursiveTensorToStableHLO(valueV reflect.Value, shape shapes.Shape, flatIdx, axis int, sb *strings.Builder) int {
	sb.WriteString("[")
	if axis == shape.Rank()-1 {
		// Case 1: the last axis we actually print the values.
		for axisIdx := range shape.Dimensions[axis] {
			if axisIdx > 0 {
				sb.WriteString(", ")
			}
			sb.WriteString(podToStableHLO(valueV.Index(flatIdx).Interface()))
			flatIdx++
		}

	} else {
		// Case 2: we recursively print the sub-tensors.
		for axisIdx := range shape.Dimensions[axis] {
			if axisIdx > 0 {
				sb.WriteString(", ")
			}
			flatIdx = recursiveTensorToStableHLO(valueV, shape, flatIdx, axis+1, sb)
		}
	}
	sb.WriteString("]")
	return flatIdx
}
