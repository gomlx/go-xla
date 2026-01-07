package stablehlo

import (
	"fmt"
	"strings"
	"testing"

	"github.com/gomlx/go-xla/pkg/types/dtypes"
	"github.com/gomlx/go-xla/pkg/types/shapes"
)

func TestIf(t *testing.T) {
	t.Run("simple scalar selection", func(t *testing.T) {
		b := New(t.Name())
		fn := b.Main()

		// Predicate: true
		pred := must1(fn.ConstantFromScalar(true))

		// True branch: return 42
		trueBranch := fn.Closure()
		trueVal := must1(trueBranch.ConstantFromScalar(int32(42)))
		if err := trueBranch.Return(trueVal); err != nil {
			t.Fatalf("trueBranch.Return: %v", err)
		}

		// False branch: return 0
		falseBranch := fn.Closure()
		falseVal := must1(falseBranch.ConstantFromScalar(int32(0)))
		if err := falseBranch.Return(falseVal); err != nil {
			t.Fatalf("falseBranch.Return: %v", err)
		}

		// Execute if
		results, err := If(pred, trueBranch, falseBranch)
		if err != nil {
			t.Fatalf("If: %v", err)
		}
		if len(results) != 1 {
			t.Fatalf("expected 1 result, got %d", len(results))
		}

		// Return the result
		if err := fn.Return(results[0]); err != nil {
			t.Fatalf("fn.Return: %v", err)
		}

		program := string(must1(b.Build()))
		fmt.Printf("%s program:\n%s\n", t.Name(), program)

		// Verify the program contains the expected if structure
		if !strings.Contains(program, "stablehlo.if") {
			t.Fatal("program missing 'stablehlo.if' operation")
		}
		if !strings.Contains(program, "true_branch") {
			t.Fatal("program missing true_branch region")
		}
		if !strings.Contains(program, "false_branch") {
			t.Fatal("program missing false_branch region")
		}
	})

	t.Run("multiple return values", func(t *testing.T) {
		b := New(t.Name())
		fn := b.Main()

		// Predicate
		pred := must1(fn.ConstantFromScalar(false))

		// True branch: return (1, 2)
		trueBranch := fn.Closure()
		trueVal1 := must1(trueBranch.ConstantFromScalar(int32(1)))
		trueVal2 := must1(trueBranch.ConstantFromScalar(int32(2)))
		if err := trueBranch.Return(trueVal1, trueVal2); err != nil {
			t.Fatalf("trueBranch.Return: %v", err)
		}

		// False branch: return (10, 20)
		falseBranch := fn.Closure()
		falseVal1 := must1(falseBranch.ConstantFromScalar(int32(10)))
		falseVal2 := must1(falseBranch.ConstantFromScalar(int32(20)))
		if err := falseBranch.Return(falseVal1, falseVal2); err != nil {
			t.Fatalf("falseBranch.Return: %v", err)
		}

		// Execute if
		results, err := If(pred, trueBranch, falseBranch)
		if err != nil {
			t.Fatalf("If: %v", err)
		}
		if len(results) != 2 {
			t.Fatalf("expected 2 results, got %d", len(results))
		}

		// Return both values
		if err := fn.Return(results...); err != nil {
			t.Fatalf("fn.Return: %v", err)
		}

		program := string(must1(b.Build()))
		fmt.Printf("%s program:\n%s\n", t.Name(), program)

		// Verify the program structure
		if !strings.Contains(program, "stablehlo.if") {
			t.Fatal("program missing 'stablehlo.if' operation")
		}
	})

	t.Run("tensor values", func(t *testing.T) {
		b := New(t.Name())
		fn := b.Main()

		// Predicate
		pred := must1(fn.ConstantFromScalar(true))

		// True branch: return tensor [1, 2, 3]
		trueBranch := fn.Closure()
		trueVec := must1(trueBranch.ConstantFromFlatAndDimensions([]int32{1, 2, 3}, 3))
		if err := trueBranch.Return(trueVec); err != nil {
			t.Fatalf("trueBranch.Return: %v", err)
		}

		// False branch: return tensor [4, 5, 6]
		falseBranch := fn.Closure()
		falseVec := must1(falseBranch.ConstantFromFlatAndDimensions([]int32{4, 5, 6}, 3))
		if err := falseBranch.Return(falseVec); err != nil {
			t.Fatalf("falseBranch.Return: %v", err)
		}

		// Execute if
		results, err := If(pred, trueBranch, falseBranch)
		if err != nil {
			t.Fatalf("If: %v", err)
		}

		// Return the result
		if err := fn.Return(results[0]); err != nil {
			t.Fatalf("fn.Return: %v", err)
		}

		program := string(must1(b.Build()))
		fmt.Printf("%s program:\n%s\n", t.Name(), program)

		// Verify tensor type in output
		if !strings.Contains(program, "tensor<3xi32>") {
			t.Fatal("program missing tensor<3xi32> type")
		}
	})

	t.Run("non-bool predicate error", func(t *testing.T) {
		b := New(t.Name())
		fn := b.Main()

		// Invalid predicate: int instead of bool
		pred := must1(fn.ConstantFromScalar(int32(1)))

		// True branch
		trueBranch := fn.Closure()
		trueVal := must1(trueBranch.ConstantFromScalar(int32(42)))
		if err := trueBranch.Return(trueVal); err != nil {
			t.Fatalf("trueBranch.Return: %v", err)
		}

		// False branch
		falseBranch := fn.Closure()
		falseVal := must1(falseBranch.ConstantFromScalar(int32(0)))
		if err := falseBranch.Return(falseVal); err != nil {
			t.Fatalf("falseBranch.Return: %v", err)
		}

		// This should fail because predicate is not bool
		_, err := If(pred, trueBranch, falseBranch)
		if err == nil {
			t.Fatal("expected error for non-bool predicate, got nil")
		}
		if !strings.Contains(err.Error(), "scalar bool") {
			t.Fatalf("expected error about scalar bool, got: %v", err)
		}
	})

	t.Run("shape mismatch error", func(t *testing.T) {
		b := New(t.Name())
		fn := b.Main()

		// Predicate
		pred := must1(fn.ConstantFromScalar(true))

		// True branch: return scalar
		trueBranch := fn.Closure()
		trueVal := must1(trueBranch.ConstantFromScalar(int32(42)))
		if err := trueBranch.Return(trueVal); err != nil {
			t.Fatalf("trueBranch.Return: %v", err)
		}

		// False branch: return vector (different shape!)
		falseBranch := fn.Closure()
		falseVec := must1(falseBranch.ConstantFromFlatAndDimensions([]int32{1, 2, 3}, 3))
		if err := falseBranch.Return(falseVec); err != nil {
			t.Fatalf("falseBranch.Return: %v", err)
		}

		// This should fail because branch outputs have different shapes
		_, err := If(pred, trueBranch, falseBranch)
		if err == nil {
			t.Fatal("expected error for shape mismatch, got nil")
		}
		if !strings.Contains(err.Error(), "must be compatible") {
			t.Fatalf("expected error about shape mismatch, got: %v", err)
		}
	})

	t.Run("output count mismatch error", func(t *testing.T) {
		b := New(t.Name())
		fn := b.Main()

		// Predicate
		pred := must1(fn.ConstantFromScalar(true))

		// True branch: return 1 value
		trueBranch := fn.Closure()
		trueVal := must1(trueBranch.ConstantFromScalar(int32(42)))
		if err := trueBranch.Return(trueVal); err != nil {
			t.Fatalf("trueBranch.Return: %v", err)
		}

		// False branch: return 2 values
		falseBranch := fn.Closure()
		falseVal1 := must1(falseBranch.ConstantFromScalar(int32(1)))
		falseVal2 := must1(falseBranch.ConstantFromScalar(int32(2)))
		if err := falseBranch.Return(falseVal1, falseVal2); err != nil {
			t.Fatalf("falseBranch.Return: %v", err)
		}

		// This should fail because branches have different number of outputs
		_, err := If(pred, trueBranch, falseBranch)
		if err == nil {
			t.Fatal("expected error for output count mismatch, got nil")
		}
		if !strings.Contains(err.Error(), "same number of outputs") {
			t.Fatalf("expected error about output count, got: %v", err)
		}
	})

	t.Run("float64 values", func(t *testing.T) {
		b := New(t.Name())
		fn := b.Main()

		// Predicate
		pred := must1(fn.ConstantFromScalar(true))

		// True branch: return pi
		trueBranch := fn.Closure()
		trueVal := must1(trueBranch.ConstantFromScalar(3.14159))
		if err := trueBranch.Return(trueVal); err != nil {
			t.Fatalf("trueBranch.Return: %v", err)
		}

		// False branch: return e
		falseBranch := fn.Closure()
		falseVal := must1(falseBranch.ConstantFromScalar(2.71828))
		if err := falseBranch.Return(falseVal); err != nil {
			t.Fatalf("falseBranch.Return: %v", err)
		}

		// Execute if
		results, err := If(pred, trueBranch, falseBranch)
		if err != nil {
			t.Fatalf("If: %v", err)
		}

		// Return the result
		if err := fn.Return(results[0]); err != nil {
			t.Fatalf("fn.Return: %v", err)
		}

		program := string(must1(b.Build()))
		fmt.Printf("%s program:\n%s\n", t.Name(), program)

		// Verify float64 type
		if !strings.Contains(program, "tensor<f64>") {
			t.Fatal("program missing f64 tensor type")
		}
	})

	t.Run("branch with inputs error", func(t *testing.T) {
		b := New(t.Name())
		fn := b.Main()

		// Predicate
		pred := must1(fn.ConstantFromScalar(true))

		// True branch with an input (not allowed per StableHLO spec)
		trueBranch := fn.Closure()
		_ = must1(trueBranch.Input(shapes.Make(dtypes.Int32))) // This is not allowed
		trueVal := must1(trueBranch.ConstantFromScalar(int32(42)))
		if err := trueBranch.Return(trueVal); err != nil {
			t.Fatalf("trueBranch.Return: %v", err)
		}

		// False branch (valid)
		falseBranch := fn.Closure()
		falseVal := must1(falseBranch.ConstantFromScalar(int32(0)))
		if err := falseBranch.Return(falseVal); err != nil {
			t.Fatalf("falseBranch.Return: %v", err)
		}

		// This should fail because true branch has inputs
		_, err := If(pred, trueBranch, falseBranch)
		if err == nil {
			t.Fatal("expected error for branch with inputs, got nil")
		}
		if !strings.Contains(err.Error(), "no inputs") {
			t.Fatalf("expected error about no inputs, got: %v", err)
		}
	})
}
