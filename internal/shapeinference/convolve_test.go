package shapeinference

import (
	"strings"
	"testing"

	"github.com/gomlx/go-xla/pkg/types/shapes"
)

func TestConvolve(t *testing.T) {

	type testCase struct {
		name                               string
		input, kernel                      shapes.Shape
		inputBatch                         int
		inputChannels                      int
		inputSpatial                       []int
		kernelInputChannels                int
		kernelOutputChannels               int
		kernelSpatial                      []int
		outputBatch                        int
		outputChannels                     int
		outputSpatial                      []int
		strides                            []int
		paddings                           [][2]int
		inputDilations, kernelDilations    []int
		channelGroupCount, batchGroupCount int

		expectedError string
		output        shapes.Shape
	}
	testCases := []testCase{
		{
			name:                 "1D with padding",
			input:                S(F32, 2, 3, 5),
			kernel:               S(F32, 3, 4, 2),
			inputBatch:           0,
			inputChannels:        1,
			inputSpatial:         []int{2},
			kernelInputChannels:  0,
			kernelOutputChannels: 1,
			kernelSpatial:        []int{2},
			outputBatch:          0,
			outputChannels:       1,
			outputSpatial:        []int{2},
			strides:              []int{2},
			paddings:             [][2]int{{0, 1}},
			inputDilations:       []int{1},
			kernelDilations:      []int{1},
			channelGroupCount:    1,
			batchGroupCount:      1,

			output: S(F32, 2, 4, 3),
		},
		{
			name:                 "1D with stride 2",
			input:                S(F32, 1, 2, 6),
			kernel:               S(F32, 2, 3, 2),
			inputBatch:           0,
			inputChannels:        1,
			inputSpatial:         []int{2},
			kernelInputChannels:  0,
			kernelOutputChannels: 1,
			kernelSpatial:        []int{2},
			outputBatch:          0,
			outputChannels:       1,
			outputSpatial:        []int{2},
			strides:              []int{2},
			paddings:             [][2]int{{0, 0}},
			inputDilations:       []int{1},
			kernelDilations:      []int{1},
			channelGroupCount:    1,
			batchGroupCount:      1,

			output: S(F32, 1, 3, 3),
		},
		{
			name:                 "1D with input dilation",
			input:                S(F32, 1, 2, 4),
			kernel:               S(F32, 2, 3, 2),
			inputBatch:           0,
			inputChannels:        1,
			inputSpatial:         []int{2},
			kernelInputChannels:  0,
			kernelOutputChannels: 1,
			kernelSpatial:        []int{2},
			outputBatch:          0,
			outputChannels:       1,
			outputSpatial:        []int{2},
			strides:              []int{1},
			paddings:             [][2]int{{0, 0}},
			inputDilations:       []int{2},
			kernelDilations:      []int{1},
			channelGroupCount:    1,
			batchGroupCount:      1,

			output: S(F32, 1, 3, 6),
		},
		{
			name:                 "1D with kernel dilation",
			input:                S(F32, 1, 2, 6),
			kernel:               S(F32, 2, 3, 2),
			inputBatch:           0,
			inputChannels:        1,
			inputSpatial:         []int{2},
			kernelInputChannels:  0,
			kernelOutputChannels: 1,
			kernelSpatial:        []int{2},
			outputBatch:          0,
			outputChannels:       1,
			outputSpatial:        []int{2},
			strides:              []int{1},
			paddings:             [][2]int{{0, 0}},
			inputDilations:       []int{1},
			kernelDilations:      []int{2},
			channelGroupCount:    1,
			batchGroupCount:      1,

			output: S(F32, 1, 3, 4),
		},
		{
			name:                 "1D with feature groups",
			input:                S(F32, 1, 6, 5),
			kernel:               S(F32, 3, 4, 2),
			inputBatch:           0,
			inputChannels:        1,
			inputSpatial:         []int{2},
			kernelInputChannels:  0,
			kernelOutputChannels: 1,
			kernelSpatial:        []int{2},
			outputBatch:          0,
			outputChannels:       1,
			outputSpatial:        []int{2},
			strides:              []int{1},
			paddings:             [][2]int{{0, 0}},
			inputDilations:       []int{1},
			kernelDilations:      []int{1},
			channelGroupCount:    2,
			batchGroupCount:      1,

			output: S(F32, 1, 4, 4),
		},
		{
			name:                 "1D with batch groups",
			input:                S(F32, 4, 2, 5),
			kernel:               S(F32, 2, 4, 2),
			inputBatch:           0,
			inputChannels:        1,
			inputSpatial:         []int{2},
			kernelInputChannels:  0,
			kernelOutputChannels: 1,
			kernelSpatial:        []int{2},
			outputBatch:          0,
			outputChannels:       1,
			outputSpatial:        []int{2},
			strides:              []int{1},
			paddings:             [][2]int{{0, 0}},
			inputDilations:       []int{1},
			kernelDilations:      []int{1},
			channelGroupCount:    1,
			batchGroupCount:      2,
			output:               S(F32, 2, 4, 4),
		},
		{
			name:                 "2D convolution",
			input:                S(F32, 1, 3, 4, 4),
			kernel:               S(F32, 3, 2, 2, 2),
			inputBatch:           0,
			inputChannels:        1,
			inputSpatial:         []int{2, 3},
			kernelInputChannels:  0,
			kernelOutputChannels: 1,
			kernelSpatial:        []int{2, 3},
			outputBatch:          0,
			outputChannels:       1,
			outputSpatial:        []int{2, 3},
			strides:              []int{1, 1},
			paddings:             [][2]int{{0, 0}, {0, 0}},
			inputDilations:       []int{1, 1},
			kernelDilations:      []int{1, 1},
			channelGroupCount:    1,
			batchGroupCount:      1,

			output: S(F32, 1, 2, 3, 3),
		},
		{
			name:                 "3D convolution",
			input:                S(F32, 1, 2, 4, 4, 4),
			kernel:               S(F32, 2, 2, 2, 2, 2),
			inputBatch:           0,
			inputChannels:        1,
			inputSpatial:         []int{2, 3, 4},
			kernelInputChannels:  0,
			kernelOutputChannels: 1,
			kernelSpatial:        []int{2, 3, 4},
			outputBatch:          0,
			outputChannels:       1,
			outputSpatial:        []int{2, 3, 4},
			strides:              []int{1, 1, 1},
			paddings:             [][2]int{{0, 0}, {0, 0}, {0, 0}},
			inputDilations:       []int{1, 1, 1},
			kernelDilations:      []int{1, 1, 1},
			channelGroupCount:    1,
			batchGroupCount:      1,

			output: S(F32, 1, 2, 3, 3, 3),
		},
		{
			name:                 "2D convolution with transposed output",
			input:                S(F32, 1, 3, 4, 5),
			kernel:               S(F32, 3, 2, 2, 2),
			inputBatch:           0,
			inputChannels:        1,
			inputSpatial:         []int{2, 3},
			kernelInputChannels:  0,
			kernelOutputChannels: 1,
			kernelSpatial:        []int{2, 3},
			outputBatch:          2,
			outputChannels:       0,
			outputSpatial:        []int{3, 1},
			strides:              []int{1, 1},
			paddings:             [][2]int{{0, 0}, {0, 0}},
			inputDilations:       []int{1, 1},
			kernelDilations:      []int{1, 1},
			channelGroupCount:    1,
			batchGroupCount:      1,

			output: S(F32, 2, 4, 1, 3),
		},
	}
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			output, err := Convolve(tc.input, tc.kernel,
				tc.strides, tc.paddings, tc.inputDilations, tc.kernelDilations,
				tc.inputBatch, tc.inputChannels, tc.inputSpatial,
				tc.kernelInputChannels, tc.kernelOutputChannels, tc.kernelSpatial,
				tc.outputBatch, tc.outputChannels, tc.outputSpatial,
				tc.channelGroupCount, tc.batchGroupCount)
			if tc.expectedError != "" {
				if err == nil {
					t.Fatalf("expected error containing %q, got nil", tc.expectedError)
				}
				if !strings.Contains(err.Error(), tc.expectedError) {
					t.Fatalf("expected error containing %q, got %q", tc.expectedError, err.Error())
				}
				return
			}
			if err != nil {
				t.Fatalf("expected no error, got %v", err)
			}
			if !tc.output.Equal(output) {
				t.Errorf("expected output %v, got %v", tc.output, output)
			}
		})
	}
}
