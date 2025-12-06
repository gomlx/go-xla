package shapes

import (
	"fmt"
	"reflect"

	"github.com/gomlx/gopjrt/dtypes"
	"github.com/pkg/errors"
)

// FromAnyValue attempts to convert a Go "any" value to its expected shape.
// Accepted values are plain-old-data (POD) types (ints, floats, complex), slices (or multiple level of slices) of POD.
//
// It returns the expected shape.
//
// Example:
//
//	shape := shapes.FromAnyValue([][]float64{{0, 0}}) // Returns shape (Float64)[1 2]
func FromAnyValue(v any) (shape Shape, err error) {
	err = shapeForAnyValueRecursive(&shape, reflect.ValueOf(v), reflect.TypeOf(v))
	return
}

func shapeForAnyValueRecursive(shape *Shape, v reflect.Value, t reflect.Type) error {
	if t.Kind() != reflect.Slice {
		// If it's not a slice, it must be one of the supported scalar types.
		shape.DType = dtypes.FromGoType(t)
		if shape.DType == dtypes.InvalidDType {
			return errors.Errorf("cannot convert type %q to a valid GoMLX shape (maybe type not supported yet?)", t)
		}
		return nil
	}

	// Slice: recurse into its element type (again slices or a supported POD).
	t = t.Elem()
	shape.Dimensions = append(shape.Dimensions, v.Len())
	shapePrefix := shape.Clone()

	// The first element is the reference
	if v.Len() == 0 {
		return errors.Errorf("value with empty slice not valid for shape conversion: %T: %v -- it wouldn't be possible to figure out the inner dimensions", v.Interface(), v)
	}
	v0 := v.Index(0)
	err := shapeForAnyValueRecursive(shape, v0, t)
	if err != nil {
		return err
	}

	// Test that other elements have the same shape as the first one.
	for ii := 1; ii < v.Len(); ii++ {
		shapeTest := shapePrefix.Clone()
		err = shapeForAnyValueRecursive(&shapeTest, v.Index(ii), t)
		if err != nil {
			return err
		}
		if !shape.Equal(shapeTest) {
			return fmt.Errorf("sub-slices have irregular shapes, found shapes %q, and %q", shape, shapeTest)
		}
	}
	return nil
}
