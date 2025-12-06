/*
 *	Copyright 2023 Jan Pfeifer
 *
 *	Licensed under the Apache License, Version 2.0 (the "License");
 *	you may not use this file except in compliance with the License.
 *	You may obtain a copy of the License at
 *
 *	http://www.apache.org/licenses/LICENSE-2.0
 *
 *	Unless required by applicable law or agreed to in writing, software
 *	distributed under the License is distributed on an "AS IS" BASIS,
 *	WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *	See the License for the specific language governing permissions and
 *	limitations under the License.
 */

package shapes

import (
	"reflect"
	"testing"

	"github.com/gomlx/gopjrt/dtypes"
)

func TestCastAsDType(t *testing.T) {
	value := [][]int{{1, 2}, {3, 4}, {5, 6}}
	{
		want := [][]float32{{1, 2}, {3, 4}, {5, 6}}
		got := CastAsDType(value, dtypes.Float32)
		if !reflect.DeepEqual(want, got) {
			t.Errorf("CastAsDType(..., Float32) = %v, want %v", got, want)
		}
	}
	{
		want := [][]complex64{{1, 2}, {3, 4}, {5, 6}}
		got := CastAsDType(value, dtypes.Complex64)
		if !reflect.DeepEqual(want, got) {
			t.Errorf("CastAsDType(..., Complex64) = %v, want %v", got, want)
		}
	}
}

func TestShape(t *testing.T) {
	invalidShape := Invalid()
	if invalidShape.Ok() {
		t.Error("Invalid().Ok() should be false")
	}

	shape0 := Make(dtypes.Float64)
	if !shape0.Ok() {
		t.Error("shape0.Ok() should be true")
	}
	if !shape0.IsScalar() {
		t.Error("shape0.IsScalar() should be true")
	}
	if shape0.IsTuple() {
		t.Error("shape0.IsTuple() should be false")
	}
	if shape0.Rank() != 0 {
		t.Errorf("shape0.Rank() = %d, want 0", shape0.Rank())
	}
	if len(shape0.Dimensions) != 0 {
		t.Errorf("len(shape0.Dimensions) = %d, want 0", len(shape0.Dimensions))
	}
	if shape0.Size() != 1 {
		t.Errorf("shape0.Size() = %d, want 1", shape0.Size())
	}
	if int(shape0.Memory()) != 8 {
		t.Errorf("shape0.Memory() = %d, want 8", int(shape0.Memory()))
	}

	shape1 := Make(dtypes.Float32, 4, 3, 2)
	if !shape1.Ok() {
		t.Error("shape1.Ok() should be true")
	}
	if shape1.IsScalar() {
		t.Error("shape1.IsScalar() should be false")
	}
	if shape1.IsTuple() {
		t.Error("shape1.IsTuple() should be false")
	}
	if shape1.Rank() != 3 {
		t.Errorf("shape1.Rank() = %d, want 3", shape1.Rank())
	}
	if len(shape1.Dimensions) != 3 {
		t.Errorf("len(shape1.Dimensions) = %d, want 3", len(shape1.Dimensions))
	}
	if shape1.Size() != 4*3*2 {
		t.Errorf("shape1.Size() = %d, want %d", shape1.Size(), 4*3*2)
	}
	if int(shape1.Memory()) != 4*4*3*2 {
		t.Errorf("shape1.Memory() = %d, want %d", int(shape1.Memory()), 4*4*3*2)
	}
}

func panics(t *testing.T, f func()) {
	t.Helper()
	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic, but code did not panic")
		}
	}()
	f()
}

func notPanics(t *testing.T, f func()) {
	t.Helper()
	defer func() {
		if r := recover(); r != nil {
			t.Errorf("expected no panic, but code panicked: %v", r)
		}
	}()
	f()
}

func TestDim(t *testing.T) {
	shape := Make(dtypes.Float32, 4, 3, 2)
	if d := shape.Dim(0); d != 4 {
		t.Errorf("shape.Dim(0) = %d, want 4", d)
	}
	if d := shape.Dim(1); d != 3 {
		t.Errorf("shape.Dim(1) = %d, want 3", d)
	}
	if d := shape.Dim(2); d != 2 {
		t.Errorf("shape.Dim(2) = %d, want 2", d)
	}
	if d := shape.Dim(-3); d != 4 {
		t.Errorf("shape.Dim(-3) = %d, want 4", d)
	}
	if d := shape.Dim(-2); d != 3 {
		t.Errorf("shape.Dim(-2) = %d, want 3", d)
	}
	if d := shape.Dim(-1); d != 2 {
		t.Errorf("shape.Dim(-1) = %d, want 2", d)
	}
	panics(t, func() { _ = shape.Dim(3) })
	panics(t, func() { _ = shape.Dim(-4) })
}

func TestFromAnyValue(t *testing.T) {
	shape, err := FromAnyValue([]int32{1, 2, 3})
	if err != nil {
		t.Fatalf("FromAnyValue failed: %v", err)
	}
	notPanics(t, func() {
		if err := shape.Check(dtypes.Int32, 3); err != nil {
			panic(err)
		}
	})

	shape, err = FromAnyValue([][][]complex64{{{1, 2, -3}, {3, 4 + 2i, -7 - 1i}}})
	if err != nil {
		t.Fatalf("FromAnyValue failed: %v", err)
	}
	notPanics(t, func() {
		if err := shape.Check(dtypes.Complex64, 1, 2, 3); err != nil {
			panic(err)
		}
	})

	// Irregular shape is not accepted:
	shape, err = FromAnyValue([][]float32{{1, 2, 3}, {4, 5}})
	if err == nil {
		t.Errorf("irregular shape should have returned an error, instead got shape %s", shape)
	}
}
