package shapes

import (
	"fmt"
	"slices"
	"strings"

	"github.com/gomlx/go-xla/pkg/types/dtypes"
)

// Quantization represents the metadata for a !quant.uniform type.
//
// Note: **experimental**, it is not supported by standard CPU PJRT.
type Quantization struct {
	// StorageType is the integer type used in memory (e.g., dtypes.Int8, dtypes.Uint4).
	StorageType dtypes.DType

	// ExpressedType is the original floating-point range being represented (e.g., dtypes.Float32).
	ExpressedType dtypes.DType

	// Scales are the step sizes.
	// - Per-tensor: 1 value.
	// - Per-axis: N values (matching the dimension size).
	// - Blockwise: M values (matching the number of blocks).
	Scales []float64

	// ZeroPoints are the integer values representing the real 0.0.
	// Must have the same length as Scales.
	ZeroPoints []int64

	// QuantizedAxes defines which axes have specific quantization parameters.
	// - Per-tensor: Empty.
	// - Per-axis: Exactly 1 axis index.
	// - Blockwise: Multiple axis indices.
	QuantizedAxes []int

	// BlockSizes defines the size of blocks along the QuantizedAxes.
	// If empty but QuantizedAxes is not, it implies standard per-axis (block size = 1).
	BlockSizes []int64
}

// UniformQuantization returns a new Quantization metadata for per-tensor (global) quantization.
// In this mode, a single scale and zero-point are applied to the entire tensor.
func UniformQuantization(storageType, expressedType dtypes.DType, scale float64, zeroPoint int64) *Quantization {
	return &Quantization{
		StorageType:   storageType,
		ExpressedType: expressedType,
		Scales:        []float64{scale},
		ZeroPoints:    []int64{zeroPoint},
		// QuantizedAxes and BlockSizes are left nil/empty for per-tensor mode.
	}
}

// ToStableHLO renders the Quantization type to StableHLO.
// The format is: !quant.uniform<StorageType:ExpressedType[:AxisInfo], {Params}>
func (q *Quantization) ToStableHLO() string {
	if q == nil {
		return "<nil>"
	}
	var sb strings.Builder
	w := func(format string, args ...any) {
		_, _ = fmt.Fprintf(&sb, format, args...) // Writing to stringbuffer shouldn't return an error.
	}
	w("!quant.uniform<%s:%s", q.StorageType.ToStableHLO(), q.ExpressedType.ToStableHLO())

	// 2. Axis/Block Info
	// If QuantizedAxes is provided, we add the axis metadata.
	if len(q.QuantizedAxes) > 0 {
		w(":")
		if len(q.BlockSizes) > 0 {
			// Sub-channel / Blockwise: {axis:blockSize, ...}
			w("{")
			for i, axis := range q.QuantizedAxes {
				if i > 0 {
					w(", ")
				}
				blockSize := int64(1) // Default
				if i < len(q.BlockSizes) {
					blockSize = q.BlockSizes[i]
				}
				w("%d:%d", axis, blockSize)
			}
			w("}")
		} else {
			// Standard Per-Axis: just the axis index (e.g., ":0")
			// StableHLO usually supports one quantization axis at a time for per-axis.
			w("%d", q.QuantizedAxes[0])
		}
	}
	w(", ")

	// 3. Parameters (Scales and ZeroPoints)
	if len(q.QuantizedAxes) > 0 {
		w("{")
		for i := range q.Scales {
			if i > 0 {
				w(", ")
			}
			w("%g:%d", q.Scales[i], q.ZeroPoints[i])
		}
		w("}")
	} else if len(q.Scales) == 1 {
		// Single parameter (Per-tensor): scale:zp
		w("%g:%d", q.Scales[0], q.ZeroPoints[0])
	}
	w(">")
	return sb.String()
}

// Clone returns a new deep copy of the Quantization.
func (q *Quantization) Clone() *Quantization {
	if q == nil {
		return nil
	}
	return &Quantization{
		StorageType:   q.StorageType,
		ExpressedType: q.ExpressedType,
		Scales:        slices.Clone(q.Scales),
		ZeroPoints:    slices.Clone(q.ZeroPoints),
		QuantizedAxes: slices.Clone(q.QuantizedAxes),
		BlockSizes:    slices.Clone(q.BlockSizes),
	}
}

// String representation of quantization, for now uses the StableHLO representation.
func (q *Quantization) String() string {
	return q.ToStableHLO()
}
