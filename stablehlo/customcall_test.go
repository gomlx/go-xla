package stablehlo

import (
	"strings"
	"testing"

	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/go-xla/types/shapes"
)

// TestCustomCallFMHA emits the cuDNN flash-attention forward custom_call and checks
// the rendered StableHLO matches the form JAX emits (see
// docs/specs/reference/fmha_fwd.stablehlo in lmkit-go).
func TestCustomCallFMHA(t *testing.T) {
	b := New(t.Name())
	fn := b.Main()
	qkv := shapes.Make(dtypes.BFloat16, 2, 2048, 12, 64) // [B,S,H,D]
	q := must1(fn.Input(qkv))
	k := must1(fn.Input(qkv))
	v := must1(fn.Input(qkv))

	out := shapes.Make(dtypes.BFloat16, 2, 12, 2048, 64) // [B,H,S,D]
	scratch := shapes.Make(dtypes.Uint8, 0)
	const layoutsIn = "[dense<[3, 2, 1, 0]> : tensor<4xindex>, dense<[3, 2, 1, 0]> : tensor<4xindex>, dense<[3, 2, 1, 0]> : tensor<4xindex>]"
	const layoutsOut = "[dense<[3, 1, 2, 0]> : tensor<4xindex>, dense<0> : tensor<1xindex>]"
	const cfg = `{"cudnn_fmha_backend_config":{"fmha_scale":0.125,"mask_type":"CAUSAL","is_flash_attention":true}}`

	results, err := CustomCall("__cudnn$fmhaSoftmax", CustomCallAPIVersionStatusReturning,
		cfg, layoutsIn, layoutsOut, []shapes.Shape{out, scratch}, q, k, v)
	if err != nil {
		t.Fatalf("CustomCall: %v", err)
	}
	if len(results) != 2 {
		t.Fatalf("expected 2 results, got %d", len(results))
	}
	if err := fn.Return(results[0]); err != nil {
		t.Fatalf("Return: %v", err)
	}

	program := string(must1(b.Build()))
	t.Logf("program:\n%s", program)
	for _, want := range []string{
		`"stablehlo.custom_call"`,
		`call_target_name = "__cudnn$fmhaSoftmax"`,
		`api_version = 2 : i32`,
		`fmha_scale`,
		`operand_layouts = [dense<[3, 2, 1, 0]>`,
		`result_layouts = [dense<[3, 1, 2, 0]>`,
		`-> (tensor<2x12x2048x64xbf16>, tensor<0xui8>)`,
	} {
		if !strings.Contains(program, want) {
			t.Errorf("rendered StableHLO missing %q", want)
		}
	}
}
