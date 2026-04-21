package stablehlo

import (
	"testing"

	"github.com/gomlx/go-xla/pkg/types/dtypes"
	"github.com/gomlx/go-xla/pkg/types/shapes"
	"github.com/stretchr/testify/require"
)

func TestTensorLiteral_ToStableHLO(t *testing.T) {
	tests := []struct {
		name     string
		value    any
		shape    shapes.Shape
		expected string
	}{
		{
			name:     "Scalar Float32",
			value:    []float32{42.0},
			shape:    shapes.Make(dtypes.F32),
			expected: "dense<42.0> : tensor<f32>",
		},
		{
			name:     "1D Float32",
			value:    []float32{1.0, 2.0, 3.0},
			shape:    shapes.Make(dtypes.F32, 3),
			expected: "dense<[1.0, 2.0, 3.0]> : tensor<3xf32>",
		},
		{
			name:     "2D Float32",
			value:    []float32{1.0, 2.0, 3.0, 4.0},
			shape:    shapes.Make(dtypes.F32, 2, 2),
			expected: "dense<[[1.0, 2.0], [3.0, 4.0]]> : tensor<2x2xf32>",
		},
		{
			name:     "Zero-sized tensor 1D",
			value:    []float32{},
			shape:    shapes.Make(dtypes.F32, 0),
			expected: "dense<[]> : tensor<0xf32>",
		},
		{
			name:     "Zero-sized tensor 2D",
			value:    []float32{},
			shape:    shapes.Make(dtypes.F32, 2, 0),
			expected: "dense<[[], []]> : tensor<2x0xf32>",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tl := newTensorLiteralFromFlatAndShape(tt.value, tt.shape)
			actual := tl.ToStableHLO()
			require.Equal(t, tt.expected, actual)
		})
	}
}
