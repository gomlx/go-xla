package shapes

import (
	"testing"

	"github.com/gomlx/go-xla/pkg/types/dtypes"
)

func TestToStableHLO(t *testing.T) {
	shape := Make(dtypes.Float32, 1, 10)
	if got := shape.ToStableHLO(); got != "tensor<1x10xf32>" {
		t.Errorf("ToStableHLO() = %q, want %q", got, "tensor<1x10xf32>")
	}

	// Test scalar.
	shape = Make(dtypes.Int32)
	if got := shape.ToStableHLO(); got != "tensor<i32>" {
		t.Errorf("ToStableHLO() = %q, want %q", got, "tensor<i32>")
	}

	shape = Make(dtypes.Float32, 1, 10).WithUniformQuantization(dtypes.Int8, dtypes.Float32, 0.1, 0)
	want := "tensor<1x10x!quant.uniform<i8:f32, 0.1:0>>"
	if got := shape.ToStableHLO(); got != want {
		t.Errorf("ToStableHLO() = %q, want %q", got, want)
	}
}
