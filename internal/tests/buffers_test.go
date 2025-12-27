package tests

import (
	"fmt"
	"testing"

	"github.com/gomlx/go-xla/pkg/pjrt"
	"github.com/gomlx/go-xla/pkg/stablehlo"
	"github.com/gomlx/go-xla/pkg/types/dtypes"
	"github.com/gomlx/go-xla/pkg/types/shapes"
)

// TestSubByteBufferTransfer tests transferring sub-byte dtypes (int2/uint2/int4/uint4) buffers
// by using stablehlo to convert from Int8 to the target dtype, then inspecting the raw bytes.
func TestSubByteBufferTransfer(t *testing.T) {
	iterateClientsAndTest(t, func(t *testing.T, client *pjrt.Client) {
		// Test Int8 -> Int2 conversion
		t.Run("Int8_to_Int2", func(t *testing.T) {
			// Create input: [8]int8 with values that fit in int2 range (-2 to 1)
			inputValues := []int8{-2, -1, 0, 1, -2, 0, 1, -1}
			inputShape := shapes.Make(dtypes.Int8, 8)

			// Build StableHLO computation: f(x: int8[8]) -> int2[8] = convert(x, int2)
			builder := stablehlo.New("int8_to_int2")
			mainFn := builder.Main()
			x, err := mainFn.NamedInput("x", inputShape)
			requireNoError(t, err, "Failed to create input")

			// Convert Int8 to Int2
			converted, err := stablehlo.Convert(x, dtypes.Int2)
			requireNoError(t, err, "Failed to convert Int8 to Int2")

			err = mainFn.Return(converted)
			requireNoError(t, err, "Failed to set return value")

			compBytes, err := builder.Build()
			requireNoError(t, err, "Failed to build StableHLO")
			fmt.Printf("StableHLO:\n%s\n", string(compBytes))

			// Compile
			loadedExec, err := client.Compile().WithStableHLO(compBytes).Done()
			requireNoError(t, err, "Failed to compile")
			defer func() {
				err := loadedExec.Destroy()
				requireNoError(t, err)
			}()

			// Transfer input to device
			inputBuffer, err := pjrt.ArrayToBuffer(client, inputValues)
			requireNoError(t, err, "Failed to create input buffer: %+v", err)
			defer func() {
				err := inputBuffer.Destroy()
				requireNoError(t, err)
			}()

			// Execute
			outputBuffers, err := loadedExec.Execute(inputBuffer).DonateNone().Done()
			requireNoError(t, err, "Failed to execute")
			if len(outputBuffers) != 1 {
				t.Fatalf("Expected 1 output buffer, got %d", len(outputBuffers))
			}
			outputBuffer := outputBuffers[0]
			defer func() {
				err := outputBuffer.Destroy()
				requireNoError(t, err)
			}()

			// Verify output dtype and dimensions
			outputDtype, err := outputBuffer.DType()
			requireNoError(t, err)
			if outputDtype != dtypes.Int2 {
				t.Fatalf("Expected output dtype Int2, got %v", outputDtype)
			}

			outputDims, err := outputBuffer.Dimensions()
			requireNoError(t, err)
			expectedDims := []int{8}
			if len(outputDims) != len(expectedDims) || outputDims[0] != expectedDims[0] {
				t.Fatalf("Expected output dimensions %v, got %v", expectedDims, outputDims)
			}

			// Get the size in bytes
			sizeBytes, err := outputBuffer.Size()
			requireNoError(t, err)
			fmt.Printf("Input values: %v\n", inputValues)
			fmt.Printf("Output dtype: %v, dimensions: %v\n", outputDtype, outputDims)
			fmt.Printf("Output buffer size: %d bytes (expected: %d bytes for 8 int2 values)\n", sizeBytes, (8*2+7)/8)

			// Transfer back to host as raw bytes
			outputData := make([]byte, sizeBytes)
			err = outputBuffer.ToHost(outputData)
			requireNoError(t, err, "Failed to transfer buffer to host")

			fmt.Printf("Raw output bytes: %v\n", outputData)
			fmt.Printf("Raw output bytes (hex): %x\n", outputData)

			// Print bit-by-bit analysis
			for i, b := range outputData {
				fmt.Printf("Byte %d: 0x%02x = ", i, b)
				for bit := 0; bit < 8; bit++ {
					if (b>>bit)&1 != 0 {
						fmt.Printf("1")
					} else {
						fmt.Printf("0")
					}
					if bit == 1 || bit == 3 || bit == 5 {
						fmt.Printf(" ")
					}
				}
				fmt.Printf("\n")
			}

			// Try to unpack assuming 4 values per byte (int2)
			fmt.Printf("\nUnpacked values (assuming 4 int2 per byte):\n")
			unpackedValues := make([]int8, len(inputValues))
			for i := 0; i < len(outputData) && i*4 < len(inputValues); i++ {
				b := outputData[i]
				for j := 0; j < 4 && i*4+j < len(inputValues); j++ {
					bitOffset := j * 2
					val := (b >> bitOffset) & 0x3
					// Convert from unsigned to signed using two's complement
					var signedVal int8
					if val >= 2 {
						signedVal = int8(val) - 4
					} else {
						signedVal = int8(val)
					}
					idx := i*4 + j
					unpackedValues[idx] = signedVal
					fmt.Printf("  outputData[%d][%d] = %d (bits %d-%d: 0x%x), expected: %d\n", i, j, signedVal, bitOffset, bitOffset+1, val, inputValues[idx])
				}
			}
			fmt.Printf("\nUnpacked: %v\n", unpackedValues)
			fmt.Printf("Expected: %v\n", inputValues)
		})

		// Test Int8 -> Int4 conversion
		t.Run("Int8_to_Int4", func(t *testing.T) {
			// Create input: [8]int8 with values that fit in int4 range (-8 to 7)
			inputValues := []int8{-8, -4, 0, 4, 7, -1, 3, -7}
			inputShape := shapes.Make(dtypes.Int8, 8)

			// Build StableHLO computation: f(x: int8[8]) -> int4[8] = convert(x, int4)
			builder := stablehlo.New("int8_to_int4")
			mainFn := builder.Main()
			x, err := mainFn.NamedInput("x", inputShape)
			requireNoError(t, err, "Failed to create input")

			// Convert Int8 to Int4
			converted, err := stablehlo.Convert(x, dtypes.Int4)
			requireNoError(t, err, "Failed to convert Int8 to Int4")

			err = mainFn.Return(converted)
			requireNoError(t, err, "Failed to set return value")

			compBytes, err := builder.Build()
			requireNoError(t, err, "Failed to build StableHLO")
			fmt.Printf("StableHLO:\n%s\n", string(compBytes))

			// Compile
			loadedExec, err := client.Compile().WithStableHLO(compBytes).Done()
			requireNoError(t, err, "Failed to compile")
			defer func() {
				err := loadedExec.Destroy()
				requireNoError(t, err)
			}()

			// Transfer input to device
			inputBuffer, err := pjrt.ArrayToBuffer(client, inputValues)
			requireNoError(t, err, "Failed to create input buffer")
			defer func() {
				err := inputBuffer.Destroy()
				requireNoError(t, err)
			}()

			// Execute
			outputBuffers, err := loadedExec.Execute(inputBuffer).DonateNone().Done()
			requireNoError(t, err, "Failed to execute")
			if len(outputBuffers) != 1 {
				t.Fatalf("Expected 1 output buffer, got %d", len(outputBuffers))
			}
			outputBuffer := outputBuffers[0]
			defer func() {
				err := outputBuffer.Destroy()
				requireNoError(t, err)
			}()

			// Verify output dtype and dimensions
			outputDtype, err := outputBuffer.DType()
			requireNoError(t, err)
			if outputDtype != dtypes.Int4 {
				t.Fatalf("Expected output dtype Int4, got %v", outputDtype)
			}

			outputDims, err := outputBuffer.Dimensions()
			requireNoError(t, err)
			expectedDims := []int{8}
			if len(outputDims) != len(expectedDims) || outputDims[0] != expectedDims[0] {
				t.Fatalf("Expected output dimensions %v, got %v", expectedDims, outputDims)
			}

			// Get the size in bytes
			sizeBytes, err := outputBuffer.Size()
			requireNoError(t, err)
			fmt.Printf("Input values: %v\n", inputValues)
			fmt.Printf("Output dtype: %v, dimensions: %v\n", outputDtype, outputDims)
			fmt.Printf("Output buffer size: %d bytes (expected: %d bytes for 8 int4 values)\n", sizeBytes, (8*4+7)/8)

			// Transfer back to host as raw bytes
			outputData := make([]byte, sizeBytes)
			err = outputBuffer.ToHost(outputData)
			requireNoError(t, err, "Failed to transfer buffer to host")

			fmt.Printf("Raw output bytes: %v\n", outputData)
			fmt.Printf("Raw output bytes (hex): %x\n", outputData)

			// Print bit-by-bit analysis
			for i, b := range outputData {
				fmt.Printf("Byte %d: 0x%02x = ", i, b)
				for bit := 0; bit < 8; bit++ {
					if (b>>bit)&1 != 0 {
						fmt.Printf("1")
					} else {
						fmt.Printf("0")
					}
					if bit == 3 {
						fmt.Printf(" ")
					}
				}
				fmt.Printf("\n")
			}

			// Try to unpack assuming 2 values per byte (int4)
			fmt.Printf("\nUnpacked values (assuming 2 int4 per byte):\n")
			unpackedValues := make([]int8, len(inputValues))
			for i := 0; i < len(outputData) && i*2 < len(inputValues); i++ {
				b := outputData[i]
				for j := 0; j < 2 && i*2+j < len(inputValues); j++ {
					bitOffset := j * 4
					val := (b >> bitOffset) & 0xf
					// Convert from unsigned to signed using two's complement
					var signedVal int8
					if val >= 8 {
						signedVal = int8(val) - 16
					} else {
						signedVal = int8(val)
					}
					idx := i*2 + j
					unpackedValues[idx] = signedVal
					fmt.Printf("  outputData[%d][%d] = %d (bits %d-%d: 0x%x), expected: %d\n", i, j, signedVal, bitOffset, bitOffset+3, val, inputValues[idx])
				}
			}
			fmt.Printf("\nUnpacked: %v\n", unpackedValues)
			fmt.Printf("Expected: %v\n", inputValues)
		})
	})
}
