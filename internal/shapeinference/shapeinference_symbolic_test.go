package shapeinference

import (
	"testing"

	"github.com/gomlx/go-xla/pkg/types/dtypes"
	"github.com/gomlx/go-xla/pkg/types/shapes"
)

// TestSelectWithSymbolicDimensions verifies that Select handles symbolic dimensions correctly.
// This test addresses the issue where shapes like [1,1,1] and [-3,-3,-3] should be compatible.
func TestSelectWithSymbolicDimensions(t *testing.T) {
	tests := []struct {
		name        string
		pred        shapes.Shape
		onTrue      shapes.Shape
		onFalse     shapes.Shape
		shouldError bool
	}{
		{
			name: "static shapes (exact match)",
			pred: shapes.Shape{
				DType:      dtypes.Bool,
				Dimensions: []int{1, 1, 1},
			},
			onTrue: shapes.Shape{
				DType:      dtypes.Float32,
				Dimensions: []int{1, 1, 1},
			},
			onFalse: shapes.Shape{
				DType:      dtypes.Float32,
				Dimensions: []int{1, 1, 1},
			},
			shouldError: false,
		},
		{
			name: "symbolic pred with static values",
			pred: shapes.Shape{
				DType:      dtypes.Bool,
				Dimensions: []int{-3, -3, -3}, // symbolic dimensions
			},
			onTrue: shapes.Shape{
				DType:      dtypes.Float32,
				Dimensions: []int{1, 1, 1},
			},
			onFalse: shapes.Shape{
				DType:      dtypes.Float32,
				Dimensions: []int{1, 1, 1},
			},
			shouldError: false, // This is the key test case
		},
		{
			name: "static pred with symbolic values",
			pred: shapes.Shape{
				DType:      dtypes.Bool,
				Dimensions: []int{1, 1, 1},
			},
			onTrue: shapes.Shape{
				DType:      dtypes.Float32,
				Dimensions: []int{-3, -3, -3},
			},
			onFalse: shapes.Shape{
				DType:      dtypes.Float32,
				Dimensions: []int{-3, -3, -3},
			},
			shouldError: false,
		},
		{
			name: "all symbolic dimensions",
			pred: shapes.Shape{
				DType:      dtypes.Bool,
				Dimensions: []int{-3, -3, -3},
			},
			onTrue: shapes.Shape{
				DType:      dtypes.Float32,
				Dimensions: []int{-3, -3, -3},
			},
			onFalse: shapes.Shape{
				DType:      dtypes.Float32,
				Dimensions: []int{-3, -3, -3},
			},
			shouldError: false,
		},
		{
			name: "mixed symbolic and static (compatible)",
			pred: shapes.Shape{
				DType:      dtypes.Bool,
				Dimensions: []int{-3, 5, -3},
			},
			onTrue: shapes.Shape{
				DType:      dtypes.Float32,
				Dimensions: []int{10, 5, 20},
			},
			onFalse: shapes.Shape{
				DType:      dtypes.Float32,
				Dimensions: []int{10, 5, 20},
			},
			shouldError: false,
		},
		{
			name: "incompatible static dimensions",
			pred: shapes.Shape{
				DType:      dtypes.Bool,
				Dimensions: []int{1, 1, 1},
			},
			onTrue: shapes.Shape{
				DType:      dtypes.Float32,
				Dimensions: []int{1, 1, 1},
			},
			onFalse: shapes.Shape{
				DType:      dtypes.Float32,
				Dimensions: []int{2, 2, 2}, // different static dims
			},
			shouldError: true,
		},
		{
			name: "different ranks",
			pred: shapes.Shape{
				DType:      dtypes.Bool,
				Dimensions: []int{1, 1},
			},
			onTrue: shapes.Shape{
				DType:      dtypes.Float32,
				Dimensions: []int{1, 1, 1},
			},
			onFalse: shapes.Shape{
				DType:      dtypes.Float32,
				Dimensions: []int{1, 1, 1},
			},
			shouldError: true,
		},
		{
			name: "scalar pred",
			pred: shapes.Shape{
				DType:      dtypes.Bool,
				Dimensions: []int{},
			},
			onTrue: shapes.Shape{
				DType:      dtypes.Float32,
				Dimensions: []int{1, 1, 1},
			},
			onFalse: shapes.Shape{
				DType:      dtypes.Float32,
				Dimensions: []int{1, 1, 1},
			},
			shouldError: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			output, err := Select(tt.pred, tt.onTrue, tt.onFalse)

			if tt.shouldError {
				if err == nil {
					t.Errorf("Expected error but got none, output: %v", output)
				}
			} else {
				if err != nil {
					t.Errorf("Unexpected error: %v", err)
				}
				// Verify output shape matches onTrue
				if !output.Equal(tt.onTrue) {
					t.Errorf("Output shape %v does not match onTrue shape %v", output, tt.onTrue)
				}
			}
		})
	}
}

// TestShapesCompatible tests the shapesCompatible helper function
func TestShapesCompatible(t *testing.T) {
	tests := []struct {
		name       string
		a          shapes.Shape
		b          shapes.Shape
		compatible bool
	}{
		{
			name: "exact match",
			a: shapes.Shape{
				DType:      dtypes.Float32,
				Dimensions: []int{1, 2, 3},
			},
			b: shapes.Shape{
				DType:      dtypes.Float32,
				Dimensions: []int{1, 2, 3},
			},
			compatible: true,
		},
		{
			name: "symbolic matches static",
			a: shapes.Shape{
				DType:      dtypes.Float32,
				Dimensions: []int{-3, -3, -3},
			},
			b: shapes.Shape{
				DType:      dtypes.Float32,
				Dimensions: []int{1, 2, 3},
			},
			compatible: true,
		},
		{
			name: "static matches symbolic",
			a: shapes.Shape{
				DType:      dtypes.Float32,
				Dimensions: []int{1, 2, 3},
			},
			b: shapes.Shape{
				DType:      dtypes.Float32,
				Dimensions: []int{-3, -3, -3},
			},
			compatible: true,
		},
		{
			name: "mixed symbolic and static",
			a: shapes.Shape{
				DType:      dtypes.Float32,
				Dimensions: []int{-3, 2, -3},
			},
			b: shapes.Shape{
				DType:      dtypes.Float32,
				Dimensions: []int{1, 2, 3},
			},
			compatible: true,
		},
		{
			name: "incompatible static dimensions",
			a: shapes.Shape{
				DType:      dtypes.Float32,
				Dimensions: []int{1, 2, 3},
			},
			b: shapes.Shape{
				DType:      dtypes.Float32,
				Dimensions: []int{1, 5, 3},
			},
			compatible: false,
		},
		{
			name: "different dtypes",
			a: shapes.Shape{
				DType:      dtypes.Float32,
				Dimensions: []int{1, 2, 3},
			},
			b: shapes.Shape{
				DType:      dtypes.Float64,
				Dimensions: []int{1, 2, 3},
			},
			compatible: false,
		},
		{
			name: "different ranks",
			a: shapes.Shape{
				DType:      dtypes.Float32,
				Dimensions: []int{1, 2},
			},
			b: shapes.Shape{
				DType:      dtypes.Float32,
				Dimensions: []int{1, 2, 3},
			},
			compatible: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := shapesCompatible(tt.a, tt.b)
			if result != tt.compatible {
				t.Errorf("shapesCompatible(%v, %v) = %v, want %v",
					tt.a, tt.b, result, tt.compatible)
			}
		})
	}
}
