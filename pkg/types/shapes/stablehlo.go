package shapes

import (
	"fmt"
	"io"
	"strings"
)

// ToStableHLO returns the ToStableHLO representation of the shape's type.
func (s Shape) ToStableHLO() string {
	var sb strings.Builder
	_ = s.WriteStableHLO(&sb)
	return sb.String()
}

// WriteStableHLO writes the StableHLO representation of the shape's type to the given writer.
func (s Shape) WriteStableHLO(writer io.Writer) error {
	var err error
	w := func(format string, args ...any) {
		if err != nil {
			return
		}
		_, err = fmt.Fprintf(writer, format, args...)
	}

	if s.IsTuple() {
		w("tuple<")
		for i, subShape := range s.TupleShapes {
			if i > 0 {
				w(", ")
			}
			if err != nil {
				return err
			}
			err = subShape.WriteStableHLO(writer)
			if err != nil {
				return err
			}
		}
		w(">")
		return err
	}

	w("tensor<")
	if s.Rank() > 0 {
		for i, dim := range s.Dimensions {
			if i > 0 {
				w("x")
			}
			// StableHLO uses '?' for dynamic/symbolic dimensions
			if dim < 0 {
				w("?")
			} else {
				w("%d", dim)
			}
		}
		w("x")
	}
	// NOTE: Bounds encoding is disabled because XLA HLO translation doesn't support
	// dynamic_broadcast_in_dim and dynamic_reshape with bounded dynamism.
	// We use a different approach: use static shapes when extractable from constant computations.
	if s.Quantization != nil {
		w("%s", s.Quantization.ToStableHLO())
	} else {
		w("%s", s.DType.ToStableHLO())
	}
	w(">")
	return err
}
