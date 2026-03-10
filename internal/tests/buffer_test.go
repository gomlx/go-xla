package tests

import (
	"fmt"
	"slices"
	"testing"

	"github.com/gomlx/go-xla/pkg/pjrt"
	. "github.com/gomlx/go-xla/pkg/stablehlo"
	"github.com/gomlx/go-xla/pkg/types"
	"github.com/gomlx/go-xla/pkg/types/dtypes"
	"github.com/gomlx/go-xla/pkg/types/shapes"
)

// TestSubByteDTypes tests that sub-byte dtypes are handled correctly.
func TestSubByteDTypes(t *testing.T) {
	iterateClientsAndTest(t, func(t *testing.T, client *pjrt.Client) {
		t.Run("Int4-AsInputParameter", func(t *testing.T) {
			builder := New(t.Name())
			fn := builder.Main()
			{
				// Build computation graph.
				input := must1(fn.NamedInput("x", shapes.Make(dtypes.Uint8)))
				output := must1(BitcastConvert(input, dtypes.Int4))
				output = must1(Convert(output, dtypes.Int8))
				must(fn.Return(output))
			}
			// Build and compile
			program := must1(builder.Build())
			fmt.Printf("Sub-byte dtype test StableHLO:\n%s\n", string(program))
			loadedExec, err := client.Compile().WithStableHLO(program).Done()
			must(err)
			defer func() {
				must(loadedExec.Destroy())
			}()

			// Create input buffer "x".
			x, err := client.BufferFromHost().
				FromRawData([]byte{0xE1}, dtypes.Uint8, []int{1}).Done()
			if err != nil {
				t.Fatalf("Failed to transfer Int4 buffer from bytes: %+v", err)
			}
			defer func() {
				must(x.Destroy())
			}()

			// Execute with x as input.
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

		t.Run("Int4-BufferBitcast", func(t *testing.T) {
			builder := New(t.Name())
			fn := builder.Main()
			{
				// Build computation graph.
				input := must1(fn.NamedInput("x", shapes.Make(dtypes.Int4, 2)))
				output := must1(Convert(input, dtypes.Int8))
				must(fn.Return(output))
			}
			// Build and compile
			program := must1(builder.Build())
			fmt.Printf("Sub-byte dtype test StableHLO:\n%s\n", string(program))
			loadedExec, err := client.Compile().WithStableHLO(program).Done()
			must(err)
			defer func() {
				must(loadedExec.Destroy())
			}()

			// Create input buffer "x".
			x := must1(client.BufferFromHost().FromRawData([]byte{0xE1}, dtypes.Uint8, nil).Done())
			xTmp := must1(x.Bitcast(dtypes.Int4))
			x.Destroy()
			x = xTmp
			defer func() {
				must(x.Destroy())
			}()

			// Execute with x as input.
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

		t.Run("Int4-AsLiteral", func(t *testing.T) {
			if true {
				t.Skip("Skipping broken test: see https://github.com/openxla/xla/issues/38964")
			}
			builder := New(t.Name())
			fn := builder.Main()
			{
				// Build computation graph.
				input := must1(fn.ConstantFromFlatAndDimensions([]uint8{0xE1}, 1))
				input2 := must1(fn.NamedInput("x", shapes.Make(dtypes.Uint8, 1)))
				areSame := must1(Compare(input, input2, types.CompareEQ, types.CompareUnsigned))

				output := must1(BitcastConvert(input, dtypes.Int4))
				output = must1(Convert(output, dtypes.Int8))
				output2 := must1(BitcastConvert(input2, dtypes.Int4))
				output2 = must1(Convert(output2, dtypes.Int8))
				must(fn.Return(output, output2, areSame))
			}
			// Build and compile
			program := must1(builder.Build())
			fmt.Printf("Sub-byte dtype test StableHLO:\n%s\n", string(program))

			// Compile
			loadedExec, err := client.Compile().WithStableHLO(program).Done()
			must(err)
			defer func() {
				must(loadedExec.Destroy())
			}()

			// Create input buffer "x".
			x, err := client.BufferFromHost().
				FromRawData([]byte{0xE1}, dtypes.Uint8, []int{1}).Done()
			if err != nil {
				t.Fatalf("Failed to transfer Int4 buffer from bytes: %+v", err)
			}
			defer func() {
				must(x.Destroy())
			}()

			// Execute:
			outputBuffers, err := loadedExec.Execute(x).Done()
			must(err)
			defer func() {
				for _, b := range outputBuffers {
					must(b.Destroy())
				}
			}()

			// Check that the output is int8
			if len(outputBuffers) != 3 {
				t.Fatalf("expected 3 output buffer, got %d", len(outputBuffers))
			}
			output := outputBuffers[0]
			output2 := outputBuffers[1]
			areSame := outputBuffers[2]
			if outputDType := must1(output.DType()); outputDType != dtypes.Int8 {
				t.Errorf("expected output dtype to be Int8, got %v", outputDType)
			}
			areSameValues, _ := must2(pjrt.BufferToArray[bool](areSame))
			fmt.Printf("\t- areSame=%v\n", areSameValues)
			output2Values, _ := must2(pjrt.BufferToArray[int8](output2))
			fmt.Printf("\t- output2=%v\n", output2Values)

			gotFlat, gotDims := must2(pjrt.BufferToArray[int8](output))
			fmt.Printf("\t- Got %v (dims=%v)\n", gotFlat, gotDims)
			want := []int8{1, -2}
			if !slices.Equal(gotFlat, want) || !slices.Equal(gotDims, []int{2}) {
				t.Errorf("expected %v output, got %v (dimensions=%v)", want, gotFlat, gotDims)
			}
		})
	})
}
