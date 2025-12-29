package shapeinference

import (
	"testing"

	"github.com/gomlx/go-xla/pkg/types/dtypes"
	"github.com/gomlx/go-xla/pkg/types/shapes"
)

// TestSelectWithDynamicDimensions verifies that Select handles dynamic dimensions correctly.
// This test addresses the issue where shapes like [1,1,1] and [DimUnknown,DimUnknown,DimUnknown] should be compatible.
func TestSelectWithDynamicDimensions(t *testing.T) {
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
			name: "dynamic pred with static values",
			pred: shapes.Shape{
				DType:      dtypes.Bool,
				Dimensions: []int{shapes.DimUnknown, shapes.DimUnknown, shapes.DimUnknown},
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
			name: "static pred with dynamic values",
			pred: shapes.Shape{
				DType:      dtypes.Bool,
				Dimensions: []int{1, 1, 1},
			},
			onTrue: shapes.Shape{
				DType:      dtypes.Float32,
				Dimensions: []int{shapes.DimUnknown, shapes.DimUnknown, shapes.DimUnknown},
			},
			onFalse: shapes.Shape{
				DType:      dtypes.Float32,
				Dimensions: []int{shapes.DimUnknown, shapes.DimUnknown, shapes.DimUnknown},
			},
			shouldError: false,
		},
		{
			name: "all dynamic dimensions",
			pred: shapes.Shape{
				DType:      dtypes.Bool,
				Dimensions: []int{shapes.DimUnknown, shapes.DimUnknown, shapes.DimUnknown},
			},
			onTrue: shapes.Shape{
				DType:      dtypes.Float32,
				Dimensions: []int{shapes.DimUnknown, shapes.DimUnknown, shapes.DimUnknown},
			},
			onFalse: shapes.Shape{
				DType:      dtypes.Float32,
				Dimensions: []int{shapes.DimUnknown, shapes.DimUnknown, shapes.DimUnknown},
			},
			shouldError: false,
		},
		{
			name: "mixed dynamic and static (compatible)",
			pred: shapes.Shape{
				DType:      dtypes.Bool,
				Dimensions: []int{shapes.DimUnknown, 5, shapes.DimUnknown},
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
			name: "dynamic matches static",
			a: shapes.Shape{
				DType:      dtypes.Float32,
				Dimensions: []int{shapes.DimUnknown, shapes.DimUnknown, shapes.DimUnknown},
			},
			b: shapes.Shape{
				DType:      dtypes.Float32,
				Dimensions: []int{1, 2, 3},
			},
			compatible: true,
		},
		{
			name: "static matches dynamic",
			a: shapes.Shape{
				DType:      dtypes.Float32,
				Dimensions: []int{1, 2, 3},
			},
			b: shapes.Shape{
				DType:      dtypes.Float32,
				Dimensions: []int{shapes.DimUnknown, shapes.DimUnknown, shapes.DimUnknown},
			},
			compatible: true,
		},
		{
			name: "mixed dynamic and static",
			a: shapes.Shape{
				DType:      dtypes.Float32,
				Dimensions: []int{shapes.DimUnknown, 2, shapes.DimUnknown},
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
