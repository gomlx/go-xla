package tests

import (
	"fmt"
	"slices"
	"testing"

	"github.com/gomlx/go-xla/pkg/pjrt"
	. "github.com/gomlx/go-xla/pkg/stablehlo"
	"github.com/gomlx/go-xla/pkg/types/dtypes"
	"github.com/gomlx/go-xla/pkg/types/shapes"
)

// TestSubByteDTypes tests that sub-byte dtypes are handled correctly.
func TestSubByteDTypes(t *testing.T) {
	iterateClientsAndTest(t, func(t *testing.T, client *pjrt.Client) {
		builder := New(t.Name())
		fn := builder.Main()
		input := must1(fn.NamedInput("x", shapes.Make(dtypes.Int4, 2)))
		must(fn.Return(must1(Convert(input, dtypes.Int8))))

		// Build and compile
		program := must1(builder.Build())
		fmt.Printf("Sub-byte dtype test StableHLO:\n%s\n", string(program))

		// Compile
		loadedExec, err := client.Compile().WithStableHLO(program).Done()
		must(err)
		defer func() {
			must(loadedExec.Destroy())
		}()

		// Execute with no inputs (since we're using constants)
		x, err := client.BufferFromHost().
			FromRawData([]byte{0xE1}, dtypes.Int4, []int{2}).Done()
		if err != nil {
			t.Fatalf("Failed to transfer Int4 buffer from bytes: %+v", err)
		}
		defer func() {
			must(x.Destroy())
		}()
		outputBuffers, err := loadedExec.Execute(x).Done()
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
		output := outputBuffers[0]
		if outputDType := must1(output.DType()); outputDType != dtypes.Int8 {
			t.Errorf("expected output dtype to be Int8, got %v", outputDType)
		}

		gotFlat, gotDims := must2(pjrt.BufferToArray[int8](output))
		fmt.Printf("\t- Got %v (dims=%v)\n", gotFlat, gotDims)
		want := []int8{1, -2}
		if !slices.Equal(gotFlat, want) || !slices.Equal(gotDims, []int{2}) {
			t.Errorf("expected %v output, got %v (dimensions=%v)", want, gotFlat, gotDims)
		}
	})
}
