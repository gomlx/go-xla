package stablehlo

import (
	"fmt"
	"strings"
	"testing"

	"github.com/gomlx/go-xla/pkg/types"
	"github.com/gomlx/go-xla/pkg/types/dtypes"
	"github.com/gomlx/go-xla/pkg/types/shapes"
)

func TestWhile(t *testing.T) {
	t.Run("count from 0 to 10", func(t *testing.T) {
		b := New(t.Name())
		fn := b.Main()

		// Initial state: counter = 0
		counter := must(fn.ConstantFromScalar(int32(0)))

		// Condition function: counter < 10
		condFn := fn.Closure()
		condCounter := must(condFn.Input(counter.Shape()))
		limit := must(condFn.ConstantFromScalar(int32(10)))
		cond := must(Compare(condCounter, limit, types.CompareLT, types.CompareSigned))
		if err := condFn.Return(cond); err != nil {
			t.Fatalf("condFn.Return: %v", err)
		}

		// Body function: counter = counter + 1
		bodyFn := fn.Closure()
		bodyCounter := must(bodyFn.Input(counter.Shape()))
		one := must(bodyFn.ConstantFromScalar(int32(1)))
		nextCounter := must(Add(bodyCounter, one))
		if err := bodyFn.Return(nextCounter); err != nil {
			t.Fatalf("bodyFn.Return: %v", err)
		}

		// Execute while loop
		results, err := While(condFn, bodyFn, counter)
		if err != nil {
			t.Fatalf("While: %v", err)
		}
		if len(results) != 1 {
			t.Fatalf("expected 1 result, got %d", len(results))
		}

		// Return the final counter value
		if err := fn.Return(results[0]); err != nil {
			t.Fatalf("fn.Return: %v", err)
		}

		program := string(must(b.Build()))
		fmt.Printf("%s program:\n%s\n", t.Name(), program)

		// Verify the program contains the expected while loop structure
		if !strings.Contains(program, "stablehlo.while") {
			t.Fatal("program missing 'stablehlo.while' operation")
		}
		if !strings.Contains(program, "cond") {
			t.Fatal("program missing condition region")
		}
		if !strings.Contains(program, "body") {
			t.Fatal("program missing body region")
		}
	})

	t.Run("multiple state values", func(t *testing.T) {
		b := New(t.Name())
		fn := b.Main()

		// Initial state: counter = 0, sum = 0
		counter := must(fn.ConstantFromScalar(int32(0)))
		sum := must(fn.ConstantFromScalar(int32(0)))

		// Condition function: counter < 5
		condFn := fn.Closure()
		condCounter := must(condFn.Input(counter.Shape()))
		condSum := must(condFn.Input(sum.Shape()))
		_ = condSum // not used in condition
		limit := must(condFn.ConstantFromScalar(int32(5)))
		cond := must(Compare(condCounter, limit, types.CompareLT, types.CompareSigned))
		if err := condFn.Return(cond); err != nil {
			t.Fatalf("condFn.Return: %v", err)
		}

		// Body function: counter += 1, sum += counter
		bodyFn := fn.Closure()
		bodyCounter := must(bodyFn.Input(counter.Shape()))
		bodySum := must(bodyFn.Input(sum.Shape()))
		one := must(bodyFn.ConstantFromScalar(int32(1)))
		nextCounter := must(Add(bodyCounter, one))
		nextSum := must(Add(bodySum, nextCounter))
		if err := bodyFn.Return(nextCounter, nextSum); err != nil {
			t.Fatalf("bodyFn.Return: %v", err)
		}

		// Execute while loop
		results, err := While(condFn, bodyFn, counter, sum)
		if err != nil {
			t.Fatalf("While: %v", err)
		}
		if len(results) != 2 {
			t.Fatalf("expected 2 results, got %d", len(results))
		}

		// Return both final values
		if err := fn.Return(results...); err != nil {
			t.Fatalf("fn.Return: %v", err)
		}

		program := string(must(b.Build()))
		fmt.Printf("%s program:\n%s\n", t.Name(), program)

		// Verify the program structure
		if !strings.Contains(program, "stablehlo.while") {
			t.Fatal("program missing 'stablehlo.while' operation")
		}
	})

	t.Run("shape validation", func(t *testing.T) {
		b := New(t.Name())
		fn := b.Main()

		// Initial state
		counter := must(fn.ConstantFromScalar(int32(0)))

		// Condition function with wrong output type (returns int instead of bool)
		condFn := fn.Closure()
		condCounter := must(condFn.Input(counter.Shape()))
		if err := condFn.Return(condCounter); err != nil {
			t.Fatalf("condFn.Return: %v", err)
		}

		// Body function
		bodyFn := fn.Closure()
		bodyCounter := must(bodyFn.Input(counter.Shape()))
		one := must(bodyFn.ConstantFromScalar(int32(1)))
		nextCounter := must(Add(bodyCounter, one))
		if err := bodyFn.Return(nextCounter); err != nil {
			t.Fatalf("bodyFn.Return: %v", err)
		}

		// This should fail because condition doesn't return bool
		_, err := While(condFn, bodyFn, counter)
		if err == nil {
			t.Fatal("expected error for non-bool condition, got nil")
		}
		if !strings.Contains(err.Error(), "scalar bool") {
			t.Fatalf("expected error about scalar bool, got: %v", err)
		}
	})

	t.Run("body output shape mismatch", func(t *testing.T) {
		b := New(t.Name())
		fn := b.Main()

		// Initial state: scalar int32
		counter := must(fn.ConstantFromScalar(int32(0)))

		// Condition function
		condFn := fn.Closure()
		condCounter := must(condFn.Input(counter.Shape()))
		limit := must(condFn.ConstantFromScalar(int32(10)))
		cond := must(Compare(condCounter, limit, types.CompareLT, types.CompareSigned))
		if err := condFn.Return(cond); err != nil {
			t.Fatalf("condFn.Return: %v", err)
		}

		// Body function with wrong output shape (returns vector instead of scalar)
		bodyFn := fn.Closure()
		bodyCounter := must(bodyFn.Input(counter.Shape()))
		// Convert to a vector [counter, counter]
		// First expand dims to make them rank-1, then concatenate
		expanded1 := must(Reshape(bodyCounter, shapes.Make(dtypes.Int32, 1)))
		expanded2 := must(Reshape(bodyCounter, shapes.Make(dtypes.Int32, 1)))
		vector := must(Concatenate(0, expanded1, expanded2))
		if err := bodyFn.Return(vector); err != nil {
			t.Fatalf("bodyFn.Return: %v", err)
		}

		// This should fail because body output shape doesn't match input
		_, err := While(condFn, bodyFn, counter)
		if err == nil {
			t.Fatal("expected error for shape mismatch, got nil")
		}
		if !strings.Contains(err.Error(), "must match") {
			t.Fatalf("expected error about shape mismatch, got: %v", err)
		}
	})

	t.Run("float64 loop", func(t *testing.T) {
		b := New(t.Name())
		fn := b.Main()

		// Initial state: x = 0.0
		x := must(fn.ConstantFromScalar(0.0))

		// Condition function: x < 1.0
		condFn := fn.Closure()
		condX := must(condFn.Input(x.Shape()))
		limit := must(condFn.ConstantFromScalar(1.0))
		cond := must(Compare(condX, limit, types.CompareLT, types.CompareFloat))
		if err := condFn.Return(cond); err != nil {
			t.Fatalf("condFn.Return: %v", err)
		}

		// Body function: x = x + 0.1
		bodyFn := fn.Closure()
		bodyX := must(bodyFn.Input(x.Shape()))
		delta := must(bodyFn.ConstantFromScalar(0.1))
		nextX := must(Add(bodyX, delta))
		if err := bodyFn.Return(nextX); err != nil {
			t.Fatalf("bodyFn.Return: %v", err)
		}

		// Execute while loop
		results, err := While(condFn, bodyFn, x)
		if err != nil {
			t.Fatalf("While: %v", err)
		}

		// Return the final value
		if err := fn.Return(results[0]); err != nil {
			t.Fatalf("fn.Return: %v", err)
		}

		program := string(must(b.Build()))
		fmt.Printf("%s program:\n%s\n", t.Name(), program)

		// Verify the program uses f64 type
		if !strings.Contains(program, "tensor<f64>") {
			t.Fatal("program missing f64 tensor type")
		}
	})

	t.Run("tensor state", func(t *testing.T) {
		b := New(t.Name())
		fn := b.Main()

		// Initial state: tensor[3] = [0, 0, 0]
		initialVec := must(fn.ConstantFromFlatAndDimensions([]int32{0, 0, 0}, 3))

		// Condition function: check if first element < 5
		condFn := fn.Closure()
		condVec := must(condFn.Input(initialVec.Shape()))
		firstElem := must(Slice(condVec, []int{0}, []int{1}, []int{1}))
		firstScalar := must(Reshape(firstElem, shapes.Make(dtypes.Int32)))
		limit := must(condFn.ConstantFromScalar(int32(5)))
		cond := must(Compare(firstScalar, limit, types.CompareLT, types.CompareSigned))
		if err := condFn.Return(cond); err != nil {
			t.Fatalf("condFn.Return: %v", err)
		}

		// Body function: add [1, 1, 1] to the vector
		bodyFn := fn.Closure()
		bodyVec := must(bodyFn.Input(initialVec.Shape()))
		ones := must(bodyFn.ConstantFromFlatAndDimensions([]int32{1, 1, 1}, 3))
		nextVec := must(Add(bodyVec, ones))
		if err := bodyFn.Return(nextVec); err != nil {
			t.Fatalf("bodyFn.Return: %v", err)
		}

		// Execute while loop
		results, err := While(condFn, bodyFn, initialVec)
		if err != nil {
			t.Fatalf("While: %v", err)
		}

		// Return the final vector
		if err := fn.Return(results[0]); err != nil {
			t.Fatalf("fn.Return: %v", err)
		}

		program := string(must(b.Build()))
		fmt.Printf("%s program:\n%s\n", t.Name(), program)

		// Verify the program uses tensor<3xi32>
		if !strings.Contains(program, "tensor<3xi32>") {
			t.Fatal("program missing tensor<3xi32> type")
		}
	})
}
