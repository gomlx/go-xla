package stablehlo

import (
	"fmt"
	"strings"
	"testing"

	"github.com/gomlx/go-xla/pkg/types/dtypes"
	"github.com/gomlx/go-xla/pkg/types/shapes"
)

func TestCall(t *testing.T) {
	t.Run("simple", func(t *testing.T) {
		b := New(t.Name())

		// Define a callee function: f(x) = x + 1
		callee := b.NewFunction("add_one")
		arg := must1(callee.Input(shapes.Make(dtypes.F32)))
		one := must1(callee.ConstantFromScalar(float32(1.0)))
		result := must1(Add(arg, one))
		if err := callee.Return(result); err != nil {
			t.Fatalf("callee.Return: %v", err)
		}

		// Define main function
		fn := b.Main()
		input := must1(fn.ConstantFromScalar(float32(41.0)))

		// Call the callee
		callResult, err := Call(callee, input)
		if err != nil {
			t.Fatalf("Call: %v", err)
		}
		if len(callResult) != 1 {
			t.Fatalf("expected 1 result from Call, got %d", len(callResult))
		}

		if err := fn.Return(callResult[0]); err != nil {
			t.Fatalf("fn.Return: %v", err)
		}

		program := string(must1(b.Build()))
		fmt.Printf("%s program:\n%s\n", t.Name(), program)

		// Verify program structure
		want := `= "func.call"(%0) { callee = @add_one } : (tensor<f32>) -> tensor<f32>`
		if !strings.Contains(program, want) {
			t.Fatalf("program missing 'func.call' operation.\nWant:\n%s\nGot:\n%s", want, program)
		}
	})

	t.Run("multi-args", func(t *testing.T) {
		b := New(t.Name())

		// Define callee: f(x, y) = (x+y, x-y)
		callee := b.NewFunction("add_sub")
		x := must1(callee.Input(shapes.Make(dtypes.F32)))
		y := must1(callee.Input(shapes.Make(dtypes.F32)))
		sum := must1(Add(x, y))
		diff := must1(Subtract(x, y))
		if err := callee.Return(sum, diff); err != nil {
			t.Fatalf("callee.Return: %v", err)
		}

		// Define main
		fn := b.Main()
		v1 := must1(fn.ConstantFromScalar(float32(10.0)))
		v2 := must1(fn.ConstantFromScalar(float32(5.0)))

		// Call
		results, err := Call(callee, v1, v2)
		if err != nil {
			t.Fatalf("Call: %v", err)
		}
		if len(results) != 2 {
			t.Fatalf("expected 2 results from Call, got %d", len(results))
		}

		if err := fn.Return(results...); err != nil {
			t.Fatalf("fn.Return: %v", err)
		}

		program := string(must1(b.Build()))
		fmt.Printf("%s program:\n%s\n", t.Name(), program)

		want := `= "func.call"(%0, %1) { callee = @add_sub } : (tensor<f32>, tensor<f32>) -> (tensor<f32>, tensor<f32>)`
		if !strings.Contains(program, want) {
			t.Fatalf("program missing 'func.call' operation.\nWant:\n%s\nGot:\n%s", want, program)
		}
	})
}
