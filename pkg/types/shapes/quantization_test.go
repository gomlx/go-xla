package shapes

import (
	"testing"

	"github.com/gomlx/go-xla/pkg/types/dtypes"
)

func TestQuantization_ToStableHLO(t *testing.T) {
	tests := []struct {
		name string
		q    *Quantization
		want string
	}{
		{
			name: "Nil Quantization",
			q:    nil,
			want: "<nil>",
		},
		{
			name: "Per-Tensor (Global Scale)",
			q: &Quantization{
				StorageType:   dtypes.Int8,
				ExpressedType: dtypes.Float32,
				Scales:        []float64{0.025},
				ZeroPoints:    []int64{0},
			},
			want: "!quant.uniform<i8:f32, 0.025:0>",
		},
		{
			name: "Per-Axis (Channel-wise)",
			q: &Quantization{
				StorageType:   dtypes.Int8,
				ExpressedType: dtypes.Float32,
				QuantizedAxes: []int{0},
				Scales:        []float64{0.1, 0.2, 0.15},
				ZeroPoints:    []int64{0, 2, -1},
			},
			want: "!quant.uniform<i8:f32:0, {0.1:0, 0.2:2, 0.15:-1}>",
		},
		{
			name: "Blockwise (Sub-channel/LLM style)",
			q: &Quantization{
				StorageType:   dtypes.Int4,
				ExpressedType: dtypes.Float32,
				QuantizedAxes: []int{1},
				BlockSizes:    []int64{32},
				Scales:        []float64{0.5, 0.6},
				ZeroPoints:    []int64{8, 8},
			},
			want: "!quant.uniform<i4:f32:{1:32}, {0.5:8, 0.6:8}>",
		},
		{
			name: "Floating Point %g formatting",
			q: &Quantization{
				StorageType:   dtypes.Int8,
				ExpressedType: dtypes.Float32,
				Scales:        []float64{0.5000000}, // Should strip trailing zeros
				ZeroPoints:    []int64{10},
			},
			want: "!quant.uniform<i8:f32, 0.5:10>",
		},
		{
			name: "Multiple Axes (Experimental)",
			q: &Quantization{
				StorageType:   dtypes.Int8,
				ExpressedType: dtypes.Float32,
				QuantizedAxes: []int{0, 1},
				BlockSizes:    []int64{1, 32},
				Scales:        []float64{0.1},
				ZeroPoints:    []int64{0},
			},
			want: "!quant.uniform<i8:f32:{0:1, 1:32}, {0.1:0}>",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := tt.q.ToStableHLO(); got != tt.want {
				t.Errorf("Quantization.ToStableHLO() = %v, want %v", got, tt.want)
			}
		})
	}
}
