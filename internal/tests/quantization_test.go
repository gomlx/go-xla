package tests

import (
	"flag"
	"fmt"
	"testing"

	"github.com/gomlx/go-xla/pkg/pjrt"
	. "github.com/gomlx/go-xla/pkg/stablehlo"
	"github.com/gomlx/go-xla/pkg/types/dtypes"
	"github.com/gomlx/go-xla/pkg/types/shapes"
)

var flagQuantization = flag.Bool("quant", false, "Include quantization tests: disabled by default since default CPU PJRT currently doesn't support it.")

// TestQuantization tests quantization support in stablehlo.
// It creates a multiplication operation, sets quantization metadata on the output,
// and then converts it to int8 storage type before returning.
//
// Test currently broken, since PJRT currently doesn't seem to include the quantization lowering passes.
func TestQuantization(t *testing.T) {
	if !*flagQuantization {
		t.Skip("Quantization tests skipped -- to enable them use -quant")
	}
	iterateClientsAndTest(t, func(t *testing.T, client *pjrt.Client) {
		builder := New("quantization_test")
		fn := builder.Main()

		// Create two constants
		quantization0 := &shapes.Quantization{
			StorageType:   dtypes.Int8,
			ExpressedType: dtypes.Float32,
			Scales:        []float64{0.1, 0.5},
			ZeroPoints:    []int64{-30, -20},
			QuantizedAxes: []int{0},
		}
		c1 := must1(fn.ConstantFromFlatAndDimensions([]float32{4.0, 15.0}, 2))
		// c2 := must1(fn.ConstantFromFlatAndDimensions([]float32{5.0, 2.0}, 2))
		c1 = must1(UniformQuantize(c1, c1.Shape().WithQuantization(quantization0)))
		// c2 = must1(UniformQuantize(c2, c2.Shape().WithQuantization(quantization0)))

		// Multiply them and update quantization metadata on the output.
		// product := must1(Multiply(c1, c2))
		// quantization1 := shapes.UniformQuantization(dtypes.Int8, dtypes.Float32, 0.2, 0)
		// product = must1(product.WithQuantization(quantization1))
		// fmt.Printf("product.shape=%s\n", product.Shape())

		// Return the quantized value (now in int8 storage format)
		must(fn.Return(must1(UniformDequantize(c1))))
		// must(fn.Return(must1(UniformDequantize(product))))

		// Build and compile
		program := must1(builder.Build())
		fmt.Printf("Quantization test StableHLO:\n%s\n", string(program))

		// Compile
		loadedExec, err := client.Compile().WithStableHLO(program).Done()
		must(err)
		defer func() {
			must(loadedExec.Destroy())
		}()

		// Execute with no inputs (since we're using constants)
		outputBuffers, err := loadedExec.Execute().Done()
		must(err)
		defer func() {
			for _, b := range outputBuffers {
				must(b.Destroy())
			}
		}()

		// Check that the output is int8
		if len(outputBuffers) != 1 {
			t.Fatalf("expected 1 output buffer, got %d", len(outputBuffers))
		}
		if outputDType := must1(outputBuffers[0].DType()); outputDType != dtypes.Int8 {
			t.Errorf("expected output dtype to be Int8, got %v", outputDType)
		}

		got := must1(pjrt.BufferToScalar[int8](outputBuffers[0]))
		if got != 60 { // 2.0 * 3.0 = 6.0, quantized with scale 0.1 and zero point 0: 6.0 / 0.1 = 60
			t.Errorf("expected 60 output, got %d", got)
		}
	})
}
