package pjrt

import (
	"fmt"
	"reflect"
	"testing"
	"unsafe"
)

func TestNamedValuesConversion(t *testing.T) {
	options := NamedValuesMap{
		"str":            "blah",
		"int64Value":     int64(7),
		"int64Array":     []int64{11, 13, 17},
		"float32":        float32(19),
		"bool":           true,
		"invalidTypeKey": complex64(1), // Type not supported.
	}

	// Check that types not-supported return error.
	cOptions, numOptions, err := options.mallocArrayPJRT_NamedValue()
	requireErrorContains(t, err, "invalidTypeKey")
	destroyPJRT_NamedValue(cOptions, numOptions)

	// Remove invalid type, and get proper conversion:
	delete(options, "invalidTypeKey")
	cOptions, numOptions, err = options.mallocArrayPJRT_NamedValue()
	requireNoError(t, err)

	// Convert back and check values.
	fmt.Printf("options=%+v\n", options)
	cArrayOptions := unsafe.Slice(cOptions, int(numOptions))
	convertedValues := pjrtNamedValuesToMap(cArrayOptions)
	for key, option := range options {
		fmt.Printf("\tconverted[%q]=%T(%#v)\n", key, convertedValues[key], convertedValues[key])
		if !reflect.DeepEqual(option, convertedValues[key]) {
			t.Fatalf("conversion failed for key %q: original value %#v, got value %#v", key, option, convertedValues[key])
		}
	}
	destroyPJRT_NamedValue(cOptions, numOptions)
}
