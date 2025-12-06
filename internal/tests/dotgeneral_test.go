package gopjrt

import (
	"fmt"
	"strings"
	"testing"

	. "github.com/gomlx/go-xla/pkg/stablehlo"
	"github.com/gomlx/go-xla/pkg/types"
	S "github.com/gomlx/go-xla/pkg/types/shapes"
	D "github.com/gomlx/gopjrt/dtypes"
	"github.com/gomlx/gopjrt/pjrt"
)

func TestDotGeneral(t *testing.T) {
	for pluginName, client := range pjrtClientsIterator(t) {
		t.Run(pluginName, func(t *testing.T) {
			testDotGeneral(t, client)
		})
	}
}

func testDotGeneral(t *testing.T, client *pjrt.Client) {
	fmt.Printf("Running test for %s:\n", client.Plugin().String())
	wantResult := []FlatAndDims{
		{[]float32{
			242, 260, 278, 296,
			899, 962, 1025, 1088,
			773, 794, 815, 836,
			2522, 2588, 2654, 2720,
			1448, 1472, 1496, 1520,
			4289, 4358, 4427, 4496,
			2267, 2294, 2321, 2348,
			6200, 6272, 6344, 6416,
			3230, 3260, 3290, 3320,
			8255, 8330, 8405, 8480,
		}, []int{5, 2, 1, 4}},
	}
	t.Run("BatchContractingCross", func(t *testing.T) {
		builder := New(t.Name())
		fn := builder.NewFunction("main")
		one := must1(fn.ConstantFromScalar(float32(1)))
		lhs := must1(fn.Iota(S.Make(D.F32, 2*3*1*5), 0))
		lhs = must1(Add(lhs, must1(BroadcastInDim(one, lhs.Shape(), nil))))
		lhs = must1(Reshape(lhs, S.Make(D.F32, 2, 3, 1, 5)))
		rhs := must1(fn.Iota(S.Make(D.F32, 5*3*2*4), 0))
		rhs = must1(Add(rhs, must1(BroadcastInDim(one, rhs.Shape(), nil))))
		rhs = must1(Reshape(rhs, S.Make(D.F32, 5, 3, 2, 4)))
		dg := must1(DotGeneral(lhs, []int{1}, []int{3, 0}, rhs, []int{1}, []int{0, 2}).Done())
		must(fn.Return(dg))
		program := must1(builder.Build())
		fmt.Printf("%s program:\n%s", t.Name(), program)
		outputs := compileAndExecute(t, client, program)
		requireBuffersEqual(t, wantResult, outputs)
	})

	t.Run("BatchContractingCross(f32)", func(t *testing.T) {
		builder := New(t.Name())
		fn := builder.NewFunction("main")
		one := must1(fn.ConstantFromScalar(float32(1)))
		lhs := must1(fn.Iota(S.Make(D.F32, 2*3*1*5), 0))
		lhs = must1(Add(lhs, must1(BroadcastInDim(one, lhs.Shape(), nil))))
		lhs = must1(Reshape(lhs, S.Make(D.F32, 2, 3, 1, 5)))
		rhs := must1(fn.Iota(S.Make(D.F32, 5*3*2*4), 0))
		rhs = must1(Add(rhs, must1(BroadcastInDim(one, rhs.Shape(), nil))))
		rhs = must1(Reshape(rhs, S.Make(D.F32, 5, 3, 2, 4)))
		dg := must1(DotGeneral(lhs, []int{1}, []int{3, 0}, rhs, []int{1}, []int{0, 2}).
			Algorithm(&types.DotGeneralAlgorithm{
				LhsPrecisionType:           types.FloatPrecisionType{DType: D.F32},
				RhsPrecisionType:           types.FloatPrecisionType{DType: D.F32},
				AccumulationType:           types.FloatPrecisionType{DType: D.F32},
				LhsComponentCount:          1,
				RhsComponentCount:          1,
				NumPrimitiveOperations:     1,
				AllowImpreciseAccumulation: false,
			}).
			Done())
		must(fn.Return(dg))
		program := must1(builder.Build())
		fmt.Printf("%s program:\n%s", t.Name(), program)
		outputs := compileAndExecute(t, client, program)
		requireBuffersEqual(t, wantResult, outputs)
	})

	if strings.Index(strings.ToUpper(client.Plugin().String()), "CUDA") != -1 {
		t.Run("BatchContractingCross(tf32)", func(t *testing.T) {
			builder := New(t.Name())
			fn := builder.NewFunction("main")
			one := must1(fn.ConstantFromScalar(float32(1)))
			lhs := must1(fn.Iota(S.Make(D.F32, 2*3*1*5), 0))
			lhs = must1(Add(lhs, must1(BroadcastInDim(one, lhs.Shape(), nil))))
			lhs = must1(Reshape(lhs, S.Make(D.F32, 2, 3, 1, 5)))
			rhs := must1(fn.Iota(S.Make(D.F32, 5*3*2*4), 0))
			rhs = must1(Add(rhs, must1(BroadcastInDim(one, rhs.Shape(), nil))))
			rhs = must1(Reshape(rhs, S.Make(D.F32, 5, 3, 2, 4)))
			dg := must1(DotGeneral(lhs, []int{1}, []int{3, 0}, rhs, []int{1}, []int{0, 2}).
				Algorithm(&types.DotGeneralAlgorithm{
					LhsPrecisionType:           types.FloatPrecisionType{TF32: true},
					RhsPrecisionType:           types.FloatPrecisionType{TF32: true},
					AccumulationType:           types.FloatPrecisionType{DType: D.F32},
					LhsComponentCount:          1,
					RhsComponentCount:          1,
					NumPrimitiveOperations:     1,
					AllowImpreciseAccumulation: false,
				}).
				Done())
			must(fn.Return(dg))
			program := must1(builder.Build())
			fmt.Printf("%s program:\n%s", t.Name(), program)
			outputs := compileAndExecute(t, client, program)
			requireBuffersEqual(t, wantResult, outputs)
		})
	}

}
