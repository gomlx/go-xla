package pjrt

import (
	"fmt"
	"math"
	"reflect"
	"strings"
	"testing"
)

// requireNoError fails the test immediately if err is not nil.
func requireNoError(t *testing.T, err error, msgAndArgs ...any) {
	t.Helper()
	if err != nil {
		if len(msgAndArgs) > 0 {
			format, ok := msgAndArgs[0].(string)
			if ok && len(msgAndArgs) > 1 {
				t.Fatalf("%s: %v", fmt.Sprintf(format, msgAndArgs[1:]...), err)
			} else {
				t.Fatalf("%v: %v", msgAndArgs[0], err)
			}
		} else {
			t.Fatalf("unexpected error: %v", err)
		}
	}
}

// requireError fails the test immediately if err is nil.
func requireError(t *testing.T, err error, msgAndArgs ...any) {
	t.Helper()
	if err == nil {
		if len(msgAndArgs) > 0 {
			format, ok := msgAndArgs[0].(string)
			if ok && len(msgAndArgs) > 1 {
				t.Fatalf("%s: expected an error but got nil", fmt.Sprintf(format, msgAndArgs[1:]...))
			} else {
				t.Fatalf("%v: expected an error but got nil", msgAndArgs[0])
			}
		} else {
			t.Fatal("expected an error but got nil")
		}
	}
}

// requireErrorContains fails if err is nil or doesn't contain the expected substring.
func requireErrorContains(t *testing.T, err error, contains string, msgAndArgs ...any) {
	t.Helper()
	if err == nil {
		if len(msgAndArgs) > 0 {
			format, ok := msgAndArgs[0].(string)
			if ok && len(msgAndArgs) > 1 {
				t.Fatalf("%s: expected an error containing %q but got nil", fmt.Sprintf(format, msgAndArgs[1:]...), contains)
			} else {
				t.Fatalf("%v: expected an error containing %q but got nil", msgAndArgs[0], contains)
			}
		} else {
			t.Fatalf("expected an error containing %q but got nil", contains)
		}
		return
	}
	if !strings.Contains(err.Error(), contains) {
		t.Fatalf("expected error to contain %q, got: %v", contains, err)
	}
}

// assertEqual fails the test immediately if expected != actual.
func assertEqual[T comparable](t *testing.T, expected, actual T, msgAndArgs ...any) {
	t.Helper()
	if expected != actual {
		msg := formatMsgAndArgs(msgAndArgs...)
		if msg != "" {
			t.Fatalf("%s: expected %v, got %v", msg, expected, actual)
		} else {
			t.Fatalf("expected %v, got %v", expected, actual)
		}
	}
}

// assertEqualSlice fails the test if the slices are not equal.
func assertEqualSlice[T comparable](t *testing.T, expected, actual []T, msgAndArgs ...any) {
	t.Helper()
	if len(expected) != len(actual) {
		msg := formatMsgAndArgs(msgAndArgs...)
		if msg != "" {
			t.Fatalf("%s: expected slice length %d, got %d", msg, len(expected), len(actual))
		} else {
			t.Fatalf("expected slice length %d, got %d", len(expected), len(actual))
		}
		return
	}
	for i := range expected {
		if expected[i] != actual[i] {
			msg := formatMsgAndArgs(msgAndArgs...)
			if msg != "" {
				t.Fatalf("%s: slices differ at index %d: expected %v, got %v", msg, i, expected[i], actual[i])
			} else {
				t.Fatalf("slices differ at index %d: expected %v, got %v", i, expected[i], actual[i])
			}
			return
		}
	}
}

// assertNotEqual fails the test immediately if expected == actual.
func assertNotEqual[T comparable](t *testing.T, expected, actual T, msgAndArgs ...any) {
	t.Helper()
	if expected == actual {
		msg := formatMsgAndArgs(msgAndArgs...)
		if msg != "" {
			t.Fatalf("%s: expected values to differ, but both are %v", msg, expected)
		} else {
			t.Fatalf("expected values to differ, but both are %v", expected)
		}
	}
}

// assertEmpty fails if the slice/map/string is not empty.
func assertEmpty[T any](t *testing.T, value []T, msgAndArgs ...any) {
	t.Helper()
	if len(value) != 0 {
		msg := formatMsgAndArgs(msgAndArgs...)
		if msg != "" {
			t.Fatalf("%s: expected empty, got length %d", msg, len(value))
		} else {
			t.Fatalf("expected empty, got length %d", len(value))
		}
	}
}

// assertNotEmpty fails if the slice is empty.
func assertNotEmpty[T any](t *testing.T, value []T, msgAndArgs ...any) {
	t.Helper()
	if len(value) == 0 {
		msg := formatMsgAndArgs(msgAndArgs...)
		if msg != "" {
			t.Fatalf("%s: expected non-empty slice", msg)
		} else {
			t.Fatal("expected non-empty slice")
		}
	}
}

// assertLen fails if the slice doesn't have the expected length.
func assertLen[T any](t *testing.T, value []T, length int, msgAndArgs ...any) {
	t.Helper()
	if len(value) != length {
		msg := formatMsgAndArgs(msgAndArgs...)
		if msg != "" {
			t.Fatalf("%s: expected length %d, got %d", msg, length, len(value))
		} else {
			t.Fatalf("expected length %d, got %d", length, len(value))
		}
	}
}

// assertZero fails if the value is not zero/nil.
func assertZero(t *testing.T, value int, msgAndArgs ...any) {
	t.Helper()
	if value != 0 {
		msg := formatMsgAndArgs(msgAndArgs...)
		if msg != "" {
			t.Fatalf("%s: expected zero, got %v", msg, value)
		} else {
			t.Fatalf("expected zero, got %v", value)
		}
	}
}

// assertTrue fails if the condition is false.
func assertTrue(t *testing.T, condition bool, msgAndArgs ...any) {
	t.Helper()
	if !condition {
		msg := formatMsgAndArgs(msgAndArgs...)
		if msg != "" {
			t.Fatalf("%s: expected true", msg)
		} else {
			t.Fatal("expected true")
		}
	}
}

// assertFalse fails if the condition is true.
func assertFalse(t *testing.T, condition bool, msgAndArgs ...any) {
	t.Helper()
	if condition {
		msg := formatMsgAndArgs(msgAndArgs...)
		if msg != "" {
			t.Fatalf("%s: expected false", msg)
		} else {
			t.Fatal("expected false")
		}
	}
}

// assertNil fails if the value is not nil.
func assertNil(t *testing.T, value any, msgAndArgs ...any) {
	t.Helper()
	if value != nil && !reflect.ValueOf(value).IsNil() {
		msg := formatMsgAndArgs(msgAndArgs...)
		if msg != "" {
			t.Fatalf("%s: expected nil, got %v", msg, value)
		} else {
			t.Fatalf("expected nil, got %v", value)
		}
	}
}

// requirePanics fails if the function doesn't panic.
func requirePanics(t *testing.T, f func(), msgAndArgs ...any) {
	t.Helper()
	defer func() {
		if r := recover(); r == nil {
			msg := formatMsgAndArgs(msgAndArgs...)
			if msg != "" {
				t.Fatalf("%s: expected panic but didn't get one", msg)
			} else {
				t.Fatal("expected panic but didn't get one")
			}
		}
	}()
	f()
}

// requireNotPanics fails if the function panics.
func requireNotPanics(t *testing.T, f func(), msgAndArgs ...any) {
	t.Helper()
	defer func() {
		if r := recover(); r != nil {
			msg := formatMsgAndArgs(msgAndArgs...)
			if msg != "" {
				t.Fatalf("%s: unexpected panic: %v", msg, r)
			} else {
				t.Fatalf("unexpected panic: %v", r)
			}
		}
	}()
	f()
}

// assertInDelta fails if the values differ by more than delta.
func assertInDelta(t *testing.T, expected, actual, delta float64, msgAndArgs ...any) {
	t.Helper()
	if math.Abs(expected-actual) > delta {
		msg := formatMsgAndArgs(msgAndArgs...)
		if msg != "" {
			t.Fatalf("%s: expected %v to be within %v of %v", msg, actual, delta, expected)
		} else {
			t.Fatalf("expected %v to be within %v of %v", actual, delta, expected)
		}
	}
}

// assertInDeltaSlice fails if any element in the slices differs by more than delta.
func assertInDeltaSlice[T ~float32 | ~float64](t *testing.T, expected, actual []T, delta float64, msgAndArgs ...any) {
	t.Helper()
	if len(expected) != len(actual) {
		msg := formatMsgAndArgs(msgAndArgs...)
		if msg != "" {
			t.Fatalf("%s: slice lengths differ: expected %d, got %d", msg, len(expected), len(actual))
		} else {
			t.Fatalf("slice lengths differ: expected %d, got %d", len(expected), len(actual))
		}
		return
	}
	for i := range expected {
		if math.Abs(float64(expected[i])-float64(actual[i])) > delta {
			msg := formatMsgAndArgs(msgAndArgs...)
			if msg != "" {
				t.Fatalf("%s: slices differ at index %d: expected %v to be within %v of %v", msg, i, actual[i], delta, expected[i])
			} else {
				t.Fatalf("slices differ at index %d: expected %v to be within %v of %v", i, actual[i], delta, expected[i])
			}
			return
		}
	}
}

// assertDeepEqual fails if the values are not deeply equal.
func assertDeepEqual(t *testing.T, expected, actual any, msgAndArgs ...any) {
	t.Helper()
	if !reflect.DeepEqual(expected, actual) {
		msg := formatMsgAndArgs(msgAndArgs...)
		if msg != "" {
			t.Fatalf("%s: expected %#v, got %#v", msg, expected, actual)
		} else {
			t.Fatalf("expected %#v, got %#v", expected, actual)
		}
	}
}

// formatMsgAndArgs formats the optional message and arguments.
func formatMsgAndArgs(msgAndArgs ...any) string {
	if len(msgAndArgs) == 0 {
		return ""
	}
	if len(msgAndArgs) == 1 {
		return fmt.Sprintf("%v", msgAndArgs[0])
	}
	format, ok := msgAndArgs[0].(string)
	if ok {
		return fmt.Sprintf(format, msgAndArgs[1:]...)
	}
	return fmt.Sprintf("%v", msgAndArgs[0])
}

