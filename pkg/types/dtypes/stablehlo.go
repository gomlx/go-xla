package dtypes

import (
	"fmt"
)

// ToStableHLO returns the StableHLO string representation of the DType.
func (dtype DType) ToStableHLO() string {
	switch dtype {
	case F64:
		return "f64"
	case F32:
		return "f32"
	case F16:
		return "f16"
	case BFloat16:
		return "bf16"
	case Int64:
		return "i64"
	case Int32:
		return "i32"
	case Int16:
		return "i16"
	case Int8:
		return "i8"
	case Int4:
		return "i4"
	case Int2:
		return "i2"
	case U64:
		return "ui64"
	case U32:
		return "ui32"
	case U16:
		return "ui16"
	case U8:
		return "ui8"
	case U4:
		return "ui4"
	case U2:
		return "ui2"
	case Bool:
		return "i1"
	case Complex64:
		return "complex<f32>"
	case Complex128:
		return "complex<f64>"
	default:
		return fmt.Sprintf("unknown_dtype<%s>", dtype.String())
	}
}
