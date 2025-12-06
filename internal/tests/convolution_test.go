package gopjrt

import (
	"fmt"
	"testing"

	. "github.com/gomlx/go-xla/pkg/stablehlo"
	"github.com/gomlx/go-xla/pkg/types"
	"github.com/gomlx/go-xla/pkg/types/shapes"
	"github.com/gomlx/gopjrt/dtypes"
	"github.com/gomlx/gopjrt/pjrt"
)

func TestConvolution(t *testing.T) {
	iterateClientsAndTest(t, testConvolution)
}

func testConvolution(t *testing.T, client *pjrt.Client) {
	t.Run("Convolve: channels first, no padding", func(t *testing.T) {
		builder := New(t.Name())
		fn := builder.Main()
		channelA := must1(fn.Iota(shapes.Make(dtypes.F32, 1, 1, 3, 3), 2))
		bFactor := must1(fn.ConstantFromScalar(float32(0.1)))
		bFactor = must1(BroadcastInDim(bFactor, channelA.Shape(), nil))
		channelB := must1(Multiply(channelA, bFactor))
		input := must1(Concatenate(1, channelA, channelB))
		kernel := must1(fn.ConstantFromScalar(float32(1)))
		kernel = must1(BroadcastInDim(kernel, shapes.Make(dtypes.F32, 1, 2, 3, 3), nil))
		spatialAxes := []int{2, 3}
		output := must1(Convolution(input, kernel,
			nil, nil, nil, nil,
			0, 1, spatialAxes,
			1, 0, spatialAxes,
			0, 1, spatialAxes,
			1, 1,
			types.DotGeneralPrecisionDefault, types.DotGeneralPrecisionDefault,
		))
		must(fn.Return(output))
		program := must1(builder.Build())
		fmt.Printf("%s program:\n%s", t.Name(), program)
		results := compileAndExecute(t, client, program)
		requireBuffersEqual(t, []FlatAndDims{{[]float32{9.9}, []int{1, 1, 1, 1}}}, results)
	})
}
