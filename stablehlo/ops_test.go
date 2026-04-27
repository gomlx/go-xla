package stablehlo

import (
	"strings"
	"testing"
)

func TestInnerMostFunction(t *testing.T) {
	b := New(t.Name())
	fn := b.Main()

	// Create a nested function (closure)
	closureFn := fn.Closure()

	// Create a double nested function
	doubleNestedFn := closureFn.Closure()

	// Values
	valFn := must1(fn.ConstantFromScalar(1.0))
	valClosure := must1(closureFn.ConstantFromScalar(1.0))
	valDoubleNested := must1(doubleNestedFn.ConstantFromScalar(1.0))

	// Sibling function (incompatible with closureFn)
	siblingFn := fn.Closure()
	valSibling := must1(siblingFn.ConstantFromScalar(1.0))

	t.Run("single operand", func(t *testing.T) {
		got, err := innerMostFunction(valFn)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if got != fn {
			t.Errorf("expected %q, got %q", fn.Name, got.Name)
		}
	})

	t.Run("same function", func(t *testing.T) {
		got, err := innerMostFunction(valFn, valFn)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if got != fn {
			t.Errorf("expected %q, got %q", fn.Name, got.Name)
		}
	})

	t.Run("parent and child", func(t *testing.T) {
		got, err := innerMostFunction(valFn, valClosure)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if got != closureFn {
			t.Errorf("expected %q (child), got %q", closureFn.Name, got.Name)
		}
	})

	t.Run("child and parent", func(t *testing.T) {
		got, err := innerMostFunction(valClosure, valFn)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if got != closureFn {
			t.Errorf("expected %q (child), got %q", closureFn.Name, got.Name)
		}
	})

	t.Run("nested hierarchy", func(t *testing.T) {
		got, err := innerMostFunction(valFn, valClosure, valDoubleNested)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if got != doubleNestedFn {
			t.Errorf("expected %q (deepest), got %q", doubleNestedFn.Name, got.Name)
		}
	})

	t.Run("incompatible siblings", func(t *testing.T) {
		_, err := innerMostFunction(valClosure, valSibling)
		if err == nil {
			t.Fatal("expected error, got nil")
		}
		if !strings.Contains(err.Error(), "incompatible functions") {
			t.Errorf("error message %q does not contain expected substring", err.Error())
		}
	})

	t.Run("incompatible parent and sibling's child", func(t *testing.T) {
		// valFn is parent of both, but if we compare valClosure and valSibling, it fails.
		// If we compare valFn and valSibling, it works (valSibling is child).
		// Comparing valDoubleNested (child of closureFn) and valSibling (child of fn).
		// Neither is ancestor.
		_, err := innerMostFunction(valDoubleNested, valSibling)
		if err == nil {
			t.Fatal("expected error, got nil")
		}
	})
}
