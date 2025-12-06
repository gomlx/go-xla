package shapes

import (
	"testing"

	"github.com/gomlx/gopjrt/dtypes"
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
}
