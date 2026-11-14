package stablehlo

import (
	"strings"
	"testing"

	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/go-xla/types/shapes"
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
			if actual != tt.expected {
				t.Errorf("got %q, want %q", actual, tt.expected)
			}
		})
	}
}

func TestConstantFromFlatSliceMutationAndValidation(t *testing.T) {
	b := New("const_mutation")
	fn := b.Main()

	if _, err := fn.ConstantFromFlatAndDimensions(nil); err == nil {
		t.Fatal("expected error for nil flat in ConstantFromFlatAndDimensions")
	}
	if _, err := fn.ConstantFromFlatAndDimensions(float32(5)); err == nil {
		t.Fatal("expected error for non-slice flat in ConstantFromFlatAndDimensions")
	}

	scratch := []float32{1, 2, 3}
	c1, err := fn.ConstantFromFlatAndDimensions(scratch, 3)
	if err != nil {
		t.Fatalf("ConstantFromFlatAndDimensions(c1) failed: %v", err)
	}
	scratch[0], scratch[1], scratch[2] = 10, 20, 30
	c2, err := fn.ConstantFromFlatAndShape(scratch, shapes.Make(dtypes.F32, 3))
	if err != nil {
		t.Fatalf("ConstantFromFlatAndShape(c2) failed: %v", err)
	}
	scratch[0] = -999

	if err := fn.Return(c1, c2); err != nil {
		t.Fatalf("Return failed: %v", err)
	}
	mlirBytes, err := b.Build()
	if err != nil {
		t.Fatalf("Build failed: %v", err)
	}
	mlir := string(mlirBytes)
	if !strings.Contains(mlir, "dense<[1.0, 2.0, 3.0]> : tensor<3xf32>") {
		t.Errorf("expected unmutated c1 literal in MLIR, got:\n%s", mlir)
	}
	if !strings.Contains(mlir, "dense<[10.0, 20.0, 30.0]> : tensor<3xf32>") {
		t.Errorf("expected unmutated c2 literal in MLIR, got:\n%s", mlir)
	}
}
