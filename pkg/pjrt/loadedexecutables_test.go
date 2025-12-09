package pjrt

import (
	"fmt"
	"testing"

	"github.com/gomlx/go-xla/pkg/stablehlo"
	"github.com/gomlx/go-xla/pkg/types/dtypes"
	"github.com/gomlx/go-xla/pkg/types/shapes"
)

func TestDonatableConfig(t *testing.T) {
	client := getPJRTClient(t)
	builder := stablehlo.New(t.Name())
	mainFn := builder.Main()

	// f(x, y, z) = x*y + z
	scalarF32 := shapes.Make(dtypes.F32)
	x := must1(mainFn.NamedInput("x", scalarF32)) // Scalar float32.
	y := must1(mainFn.NamedInput("y", scalarF32)) // Scalar float32.
	z := must1(mainFn.NamedInput("z", scalarF32)) // Scalar float32.
	fX := capture(stablehlo.Multiply(x, y)).Test(t)
	fX = capture(stablehlo.Add(fX, z)).Test(t)

	// Take program and compile.
	err := mainFn.Return(fX)
	requireNoError(t, err, "Failed to set return value")
	compBytes := capture(builder.Build()).Test(t)
	exec, err := client.Compile().WithStableHLO(compBytes).Done()
	requireNoError(t, err, "Failed to compile program")

	fmt.Println("Memory usage:")
	fmt.Printf("OnDevice: %+v\n", exec.OnDeviceMemoryUsageStats)
	fmt.Printf("OnHost: %+v\n", exec.OnHostMemoryUsageStats)

	// Test the ExecutionConfig:
	c := exec.Execute(nil, nil, nil)                          // nil values, we are not going to actually execute it.
	assertEqualSlice(t, []int{0, 1, 2}, c.nonDonatableInputs) // None of the inputs to be donated by default.
	c = c.Donate(1)                                           // Donate 1.
	assertEqualSlice(t, []int{0, 2}, c.nonDonatableInputs)
	c = c.Donate(0) // Donate 0.
	assertEqualSlice(t, []int{2}, c.nonDonatableInputs)
	c = c.Donate(0) // Donate 0 again.
	assertEqualSlice(t, []int{2}, c.nonDonatableInputs)

	err = client.Destroy()
	requireNoError(t, err, "Failed to destroy the client")
}
