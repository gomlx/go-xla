package stablehlo

import (
	"fmt"
	"io"

	"github.com/gomlx/go-xla/pkg/types/shapes"
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
	var result string
	for _, c := range name {
		if (c >= '0' && c <= '9') || (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '_' {
			result += string(c)
		} else {
			result += "_"
		}
	}
	return result
}
