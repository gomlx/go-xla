package utils

import (
	"fmt"

	"github.com/gomlx/compute/dtypes"
)

// DefaultCPUVersion is the default version of the CPU PJRT plugin to use with
// this go-xla release, against which it was tested.
// Exported in pkg/pjrt, but a copy is kept here, so `pkg/installer` can include it without pulling the whole of PJRT.
var DefaultCPUVersion = "v0.104.0"

// NormalizeIdentifier converts the name of an identifier (function name or function input parameter
// name) to a valid one: only letters, digits, and underscores are allowed.
//
// Invalid characters are replaced with underscores.
// If the name starts with a digit, it is prefixed with an underscore.
func NormalizeIdentifier(name string) string {
	if name == "" {
		return ""
	}
	result := make([]rune, 0, len(name)+1)
	if name[0] >= '0' && name[0] <= '9' {
		result = append(result, '_')
	}
	for _, r := range name {
		if (r >= 'a' && r <= 'z') || (r >= 'A' && r <= 'Z') || (r >= '0' && r <= '9') || r == '_' {
			result = append(result, r)
		} else {
			result = append(result, '_')
		}
	}
	return string(result)
}

// DTypeToStableHLO returns the StableHLO string representation of the DType.
func DTypeToStableHLO(dtype dtypes.DType) string {
	switch dtype {
	case dtypes.F64:
		return "f64"
	case dtypes.F32:
		return "f32"
	case dtypes.F16:
		return "f16"
	case dtypes.BFloat16:
		return "bf16"
	case dtypes.Int64:
		return "i64"
	case dtypes.Int32:
		return "i32"
	case dtypes.Int16:
		return "i16"
	case dtypes.Int8:
		return "i8"
	case dtypes.Int4:
		return "i4"
	case dtypes.Int2:
		return "i2"
	case dtypes.Uint64:
		return "ui64"
	case dtypes.Uint32:
		return "ui32"
	case dtypes.Uint16:
		return "ui16"
	case dtypes.Uint8:
		return "ui8"
	case dtypes.Uint4:
		return "ui4"
	case dtypes.Uint2:
		return "ui2"
	case dtypes.Bool:
		return "i1"
	case dtypes.Complex64:
		return "complex<f32>"
	case dtypes.Complex128:
		return "complex<f64>"
	default:
		return fmt.Sprintf("unknown_dtype<%s>", dtype.String())
	}
}
