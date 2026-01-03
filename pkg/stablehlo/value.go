package stablehlo

import (
	"fmt"
	"io"
	"strings"

	"github.com/gomlx/go-xla/pkg/types/shapes"
	"github.com/pkg/errors"
)

// Value represents a value in a StableHLO program, like `%0` or `%arg0`.
// These values can be inputs, outputs or intermediary values of functions.
//
// It is always associated with a function (where it's being used) and must be uniquely identified by a string with
// digits '0'-'9', 'A'-'Z', 'a'-'z' or '_'.
//
// For inlined functions (for instance, the one passed to a Reduce operation), the names cannot clash with the parent
// function name (!?). But the names can be reused in different inline functions.
//
// It also carries its shape information.
type Value struct {
	fn         *Function
	name       string
	shape      shapes.Shape
	Attributes map[string]any

	// stmt is the statement that created this value. It is nil for function input parameters.
	stmt *Statement

	// outputIndex is the index of this value in stmt.Outputs. It is only valid when stmt != nil.
	outputIndex int
}

// Shape returns the shape of the value.
func (v *Value) Shape() shapes.Shape {
	return v.shape
}

// Write writes the value in ToStableHLO text format to the given writer.
func (v *Value) Write(w io.Writer, indentation string) error {
	_ = indentation
	_, err := fmt.Fprintf(w, "%%%s", v.name)
	return err
}

// String implements fmt.Stringer.
func (v *Value) String() string {
	return "%" + v.name
}

// ConvertToValidName replaces any characters not in { "0"-"9", "a"-"z", "A-Z", "_" } to a "_",
// making it a valid name for values and function arguments.
func ConvertToValidName(name string) string {
	var result strings.Builder
	for _, c := range name {
		if (c >= '0' && c <= '9') || (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '_' {
			result.WriteString(string(c))
		} else {
			result.WriteString("_")
		}
	}
	return result.String()
}

// WithQuantization sets the quantization parameters for this value.
// It updates the quantization metadata on the value's shape, which will be reflected
// in the StableHLO output.
//
// This method can only be called on values that were created by operations (not on
// function input parameters). Parameter values cannot have their quantization changed
// as they are external inputs.
//
// Returns an error if:
//   - The value is a function input parameter (stmt == nil)
//   - The quantization parameter is nil
func (v *Value) WithQuantization(q *shapes.Quantization) (*Value, error) {
	if v.stmt == nil {
		return nil, errors.Errorf("cannot change quantization on parameter value %s (parameter values cannot have their quantization changed)", v.name)
	}
	if q == nil {
		return nil, errors.New("quantization cannot be nil")
	}

	// Update the shape's quantization
	// Note: v and stmt.Outputs[v.outputIndex] are the same Value object, so updating v.shape
	// automatically updates the shape in the statement's output as well.
	v.shape.Quantization = q

	return v, nil
}
