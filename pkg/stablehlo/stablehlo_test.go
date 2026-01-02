package stablehlo

import (
	"fmt"
	"strings"
	"testing"

	"github.com/gomlx/go-xla/pkg/types/dtypes"
	"github.com/gomlx/go-xla/pkg/types/shapes"
	"github.com/gomlx/go-xla/pkg/types/shardy"
)

func must[T any](value T, err error) T {
	if err != nil {
		panic(err)
	}
	return value
}

func TestBuilder(t *testing.T) {
	t.Run("no inputs", func(t *testing.T) {
		b := New(t.Name())
		fn := b.Main()
		c1 := must(fn.ConstantFromScalar(1.0))
		c2 := must(fn.ConstantFromScalar(2.0))
		sum := must(Add(c1, c2))
		if err := fn.Return(sum); err != nil {
			t.Fatalf("expected no error, got %v", err)
		}
		program := string(must(b.Build()))
		fmt.Printf("%s program:\n%s", t.Name(), program)
		want := `module @TestBuilder_no_inputs {
  func.func @main() -> tensor<f64> {
    %0 = "stablehlo.constant"() { value = dense<1.0> : tensor<f64> } : () -> tensor<f64>
    %1 = "stablehlo.constant"() { value = dense<2.0> : tensor<f64> } : () -> tensor<f64>
    %2 = "stablehlo.add"(%0, %1) : (tensor<f64>, tensor<f64>) -> tensor<f64>
    "stablehlo.return"(%2) : (tensor<f64>) -> ()
  }
}
`
		if program != want {
			fmt.Printf("  Failed. Wanted the following program:\n%s", want)
			t.Fatal("programs don't match")
		}
	})

	t.Run("Sharding", func(t *testing.T) {
		b := New(t.Name())
		mesh, err := shardy.NewDeviceMesh("mesh", []int{4, 2}, []string{"data", "model"})
		if err != nil {
			t.Fatalf("expected no error, got %v", err)
		}
		err = mesh.SetLogicalDeviceAssignment(7, 6, 5, 4, 3, 2, 1, 0)
		if err != nil {
			t.Fatalf("expected no error, got %v", err)
		}
		b.WithShardy(mesh)
		fn := b.Main()

		arg0 := must(fn.NamedInputWithShardingAndAttributes(
			"arg0",
			shapes.Make(dtypes.F32, 16, 128),
			b.NewShardingSpec().AddShardedAxis("data"),
			nil,
		))
		arg1 := must(fn.NamedInputWithSharding(
			"arg1",
			shapes.Make(dtypes.F32, 128, 256),
			b.NewShardingSpec().AddShardedAxis("model"),
		))

		tanh := must(Tanh(arg0))
		dot := must(Dot(tanh, arg1))
		err = fn.ReturnWithShardingAndAttributes(
			[]*Value{dot},
			[]*shardy.ShardingSpec{
				b.NewShardingSpec().AddShardedAxis("data"),
			},
			[]map[string]any{
				{"jax.result_info": "result"},
			})
		if err != nil {
			t.Fatalf("expected no error, got %v", err)
		}

		program := string(must(b.Build()))
		fmt.Printf("%s program:\n%s", t.Name(), program)
		want := `module @TestBuilder_Sharding attributes {stablehlo.num_replicas = 1,  stablehlo.num_partitions = 8} {
  sdy.mesh @mesh = <["data"=4, "model"=2], device_ids=[7, 6, 5, 4, 3, 2, 1, 0]>
  func.func @main(%arg0: tensor<16x128xf32> { sdy.sharding = #sdy.sharding<@mesh, [{"data"}, {}]> }, %arg1: tensor<128x256xf32> { sdy.sharding = #sdy.sharding<@mesh, [{"model"}, {}]> }) -> (tensor<16x256xf32> {
    jax.result_info = "result",
    sdy.sharding = #sdy.sharding<@mesh, [{"data"}, {}]>
  }) {
    %0 = "stablehlo.tanh"(%arg0) : (tensor<16x128xf32>) -> tensor<16x128xf32>
    %1 = "stablehlo.dot_general"(%0, %arg1) {
      dot_dimension_numbers = #stablehlo.dot<
  lhs_batching_dimensions = [],
  rhs_batching_dimensions = [],
  lhs_contracting_dimensions = [1],
  rhs_contracting_dimensions = [0]
>,
      precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]
    } : (tensor<16x128xf32>, tensor<128x256xf32>) -> tensor<16x256xf32>
    "stablehlo.return"(%1) : (tensor<16x256xf32>) -> ()
  }
}
`
		if want != program {
			t.Fatalf("programs don't match.\nWant:\n%s\nGot:\n%s", want, program)
		}
	})

	t.Run("with inputs", func(t *testing.T) {
		builder := New(t.Name())
		shape := shapes.Make(dtypes.Float64)
		// lhs is provided during the Main function creation, and rhs is added later.
		fn := builder.Main()
		lhs := must(fn.NamedInput("lhs", shape))
		rhs := must(fn.NamedInput("rhs", shape))
		sum := must(Add(lhs, rhs))
		if err := fn.Return(sum); err != nil {
			t.Fatalf("expected no error, got %v", err)
		}
		program := string(must(builder.Build()))
		fmt.Printf("%s program:\n%s", t.Name(), program)
		want := `module @TestBuilder_with_inputs {
  func.func @main(%lhs: tensor<f64>, %rhs: tensor<f64>) -> tensor<f64> {
    %0 = "stablehlo.add"(%lhs, %rhs) : (tensor<f64>, tensor<f64>) -> tensor<f64>
    "stablehlo.return"(%0) : (tensor<f64>) -> ()
  }
}
`
		if program != want {
			fmt.Printf("  Failed. Wanted the following program:\n%s", want)
			t.Fatal("programs don't match")
		}
	})

	t.Run("quantization", func(t *testing.T) {
		builder := New(t.Name())
		fn := builder.Main()
		// Create two constants
		c1 := must(fn.ConstantFromScalar(float32(2.0)))
		c2 := must(fn.ConstantFromScalar(float32(3.0)))
		// Multiply them
		product := must(Multiply(c1, c2))
		// Change quantization on the output
		quantization := shapes.UniformQuantization(dtypes.Int8, dtypes.Float32, 0.025, 0)
		product, err := product.WithQuantization(quantization)
		if err != nil {
			t.Fatalf("expected no error from WithQuantization, got %v", err)
		}
		if err := fn.Return(product); err != nil {
			t.Fatalf("expected no error, got %v", err)
		}
		program := string(must(builder.Build()))
		fmt.Printf("%s program:\n%s", t.Name(), program)
		want := `module @TestBuilder_quantization {
  func.func @main() -> tensor<!quant.uniform<i8:f32, 0.025:0>> {
    %0 = "stablehlo.constant"() { value = dense<2.0> : tensor<f32> } : () -> tensor<f32>
    %1 = "stablehlo.constant"() { value = dense<3.0> : tensor<f32> } : () -> tensor<f32>
    %2 = "stablehlo.multiply"(%0, %1) : (tensor<f32>, tensor<f32>) -> tensor<!quant.uniform<i8:f32, 0.025:0>>
    "stablehlo.return"(%2) : (tensor<!quant.uniform<i8:f32, 0.025:0>>) -> ()
  }
}
`
		if program != want {
			fmt.Printf("  Failed. Wanted the following program:\n%s", want)
			t.Fatal("programs don't match")
		}
	})
}

func TestBuilder_Errors(t *testing.T) {
	t.Run("no main", func(t *testing.T) {
		b := New("test_program")
		fn := b.NewFunction("not_main", nil)
		c1 := must(fn.ConstantFromScalar(1.0))
		if err := fn.Return(c1); err != nil {
			t.Fatalf("expected no error, got %v", err)
		}
		_, err := b.Build()
		if err == nil {
			t.Fatal("expected error, got nil")
		}
		if !strings.Contains(err.Error(), "program must have a main function") {
			t.Fatalf("error message %q does not contain expected substring", err.Error())
		}
	})
}

func TestNormalizeIdentifier(t *testing.T) {
	testCases := []struct {
		input, want string
	}{
		{"abc123", "abc123"},
		{"arg#2", "arg_2"},
		{"0abc", "_0abc"},
	}
	for _, tc := range testCases {
		got := NormalizeIdentifier(tc.input)
		if got != tc.want {
			t.Errorf("NormalizeIdentifier(%q) = %q, want %q", tc.input, got, tc.want)
		}
	}
}

func TestDynamicOperations(t *testing.T) {
	t.Run("GetDimensionSize", func(t *testing.T) {
		b := New(t.Name())
		fn := b.Main()

		// Create a tensor with shape [3, ?, 5] where the second dimension is dynamic
		operand := must(fn.NamedInput("operand", shapes.Make(dtypes.Float32, 3, shapes.DimUnknown, 5)))

		// Get size of dynamic dimension 1 (value only known at runtime)
		dimSize := must(GetDimensionSize(operand, 1))

		if err := fn.Return(dimSize); err != nil {
			t.Fatalf("expected no error, got %v", err)
		}

		program := string(must(b.Build()))
		fmt.Printf("%s program:\n%s", t.Name(), program)

		// Verify the operation is present in the program
		if !strings.Contains(program, "stablehlo.get_dimension_size") {
			t.Fatal("expected program to contain get_dimension_size operation")
		}
		if !strings.Contains(program, `dimension = 1`) {
			t.Fatal("expected program to specify dimension = 1")
		}
	})

}
