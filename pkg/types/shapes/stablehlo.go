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
			// StableHLO uses '?' for dynamic dimensions (DimUnknown)
			if dim == DimUnknown {
				w("?")
			} else {
				w("%d", dim)
			}
		}
		w("x")
	}
	if s.Quantization != nil {
		w("%s", s.Quantization.ToStableHLO())
	} else {
		w("%s", s.DType.ToStableHLO())
	}

	// Encode bounds only if explicitly requested via EncodeBounds flag.
	// XLA requires bounded dynamic dimensions for compilation.
	if s.EncodeBounds && s.hasBoundedDynamism() {
		w(", #stablehlo.bounds<")
		for i, dim := range s.Dimensions {
			if i > 0 {
				w(", ")
			}
			if dim == DimUnknown {
				// Dynamic dimension - check if we have a bound
				if i < len(s.DimensionBounds) && s.DimensionBounds[i] > 0 {
					w("%d", s.DimensionBounds[i])
				} else {
					w("?")
				}
			} else {
				// Static dimension - use ? (no bound needed, the dimension is already known)
				w("?")
			}
		}
		w(">")
	}
	w(">")
	return err
}
