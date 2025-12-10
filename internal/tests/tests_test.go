package tests

import (
	"flag"
	"fmt"
	"math"
	"os"
	"runtime"
	"testing"

	"github.com/gomlx/go-xla/pkg/pjrt"
	"github.com/gomlx/go-xla/pkg/stablehlo"
	"github.com/gomlx/go-xla/pkg/types/dtypes"
	"github.com/gomlx/go-xla/pkg/types/shapes"
	"k8s.io/klog/v2"
)

var (
	flagMemProfileRate  = flag.Int("mem_profile_rate", 0, "If > 0, set to runtime.MemProfileRate. The default is 512*1024.")
	flagE2ERepeatBuilds = flag.Int("e2e_repeat_builds", 1, "Number of times to repeat the end-to-end graph building and execution.")
	flagE2ERepeatExecs  = flag.Int("e2e_repeat_execs", 1, "Number of times to repeat the end-to-end graph execution (multiplied by --e2e_repeat_builds).")
)

func init() {
	klog.InitFlags(nil)
}

func TestMain(m *testing.M) {
	flag.Parse()

	if *flagMemProfileRate > 0 {
		runtime.MemProfileRate = *flagMemProfileRate
	}
	result := m.Run()

	// Force GC before the program exits and the profile is written
	pjrt.FreeAll()
	for range 20 {
		runtime.GC()
	}
	os.Exit(result)
}

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

// assertInDelta fails if the values differ by more than delta.
func assertInDelta(t *testing.T, expected, actual, delta float64) {
	t.Helper()
	if math.Abs(expected-actual) > delta {
		t.Fatalf("expected %v to be within %v of %v", actual, delta, expected)
	}
}

// TestEndToEnd builds, compiles, and executes a minimal computation f(x) = x^2 using stablehlo to build the computation,
// and pjrt to compile and execute it.
func TestEndToEnd(t *testing.T) {
	iterateClientsAndTest(t, func(t *testing.T, client *pjrt.Client) {
		// PJRT plugin and create a client.
		fmt.Printf("\t- Plugin attributes=%+v\n", client.Plugin().Attributes())

		// List devices.
		addressableDevices := client.AddressableDevices()
		fmt.Println("Addressable devices:")
		for deviceNum, device := range addressableDevices {
			hardwareID := device.LocalHardwareID()
			addressable, err := device.IsAddressable()
			requireNoError(t, err)
			desc, err := device.GetDescription()
			requireNoError(t, err)
			fmt.Printf("\tDevice #%d: hardwareID=%d, addressable=%v, description=%s\n",
				deviceNum, hardwareID, addressable, desc.DebugString())
		}

		// Default device assignment for SPMD:
		fmt.Println()
		fmt.Printf("Default device assignment for SPMD:\n")
		for i := range client.NumDevices() {
			numReplicas := i + 1
			spmdDefaultAssignment, err := client.DefaultDeviceAssignment(numReplicas, 1)
			requireNoError(t, err, "Failed to get default device assignment")
			fmt.Printf("\tWith %d devices: %v\n", numReplicas, spmdDefaultAssignment)
		}
		fmt.Println()

		for portableIdx, portableName := range []string{"fixed-device", "portable"} {
			isPortable := portableIdx == 1
			t.Run(portableName, func(t *testing.T) {
				fmt.Printf("- %s:\n", portableName)
				for range *flagE2ERepeatBuilds {
					// f(x) = x^2+1
					builder := stablehlo.New("x_times_x_plus_1") // Use valid identifier for module name
					scalarF32 := shapes.Make(dtypes.F32)         // Scalar float32 shape

					// Create a main function and define its inputs.
					mainFn := builder.Main()
					x, err := mainFn.NamedInput("x", scalarF32)
					requireNoError(t, err, "Failed to create a named input for x")

					// Build computation graph
					fX, err := stablehlo.Multiply(x, x)
					requireNoError(t, err, "Failed operation Mul")

					one, err := mainFn.ConstantFromScalar(float32(1))
					requireNoError(t, err, "Failed to create a constant for 1")

					fX, err = stablehlo.Add(fX, one)
					requireNoError(t, err, "Failed operation Add")
					err = mainFn.Return(fX) // Set the return value for the main function
					requireNoError(t, err, "Failed to set return value")

					// Get computation created.
					compBytes, err := builder.Build()
					requireNoError(t, err, "Failed to build StableHLO from ops.")
					fmt.Printf("StableHLO:\n%s\n", string(compBytes))

					// Compile program: the default compilation is "portable", meaning it can be executed by any device.
					var loadedExec *pjrt.LoadedExecutable
					var deviceAssignment []int
					if !isPortable {
						deviceAssignment = []int{0}
					}
					loadedExec, err = client.Compile().
						WithStableHLO(compBytes).
						WithDeviceAssignment(deviceAssignment).
						Done()
					requireNoError(t, err, "Failed to compile program")
					fmt.Printf("Compiled program: name=%s, #outputs=%d\n", loadedExec.Name, loadedExec.NumOutputs)

					// Portable execution should not have a fixed assignment.
					if isPortable {
						_, _, deviceAssignment, err := loadedExec.GetDeviceAssignment()
						requireNoError(t, err, "Failed to get device assignment for execution")
						if deviceAssignment != nil {
							t.Fatal("default computation is 'portable', meaning there should be no device assignment")
						}
					}

					// Test devices and values.
					inputs := []float32{0.1, 1, 3, 4, 5}
					wants := []float32{1.01, 2, 10, 17, 26}
					fmt.Printf("f(x) = x^2 + 1:\n")
					devicesToTest := client.NumDevices()
					if !isPortable {
						// Only use device 0
						devicesToTest = 1
					}
					for deviceNum := range devicesToTest {
						if devicesToTest > 1 {
							fmt.Printf(" Device #%d:\n", deviceNum)
						}
						for range *flagE2ERepeatExecs {
							for ii, input := range inputs {
								// Transfer input to an on-device buffer.
								inputBuffer, err := pjrt.ScalarToBufferOnDeviceNum(client, deviceNum, input)
								requireNoError(t, err, "Failed to create on-device buffer for input %v, deviceNum=%d", input, deviceNum)

								// Execute: it returns the output on-device buffer(s).
								var outputBuffers []*pjrt.Buffer
								if isPortable {
									outputBuffers, err = loadedExec.Execute(inputBuffer).OnDeviceByNum(deviceNum).Done()
								} else {
									outputBuffers, err = loadedExec.Execute(inputBuffer).Done()
								}
								requireNoError(t, err, "Failed to execute on input %g, deviceNum=%d", input, deviceNum)
								if len(outputBuffers) != 1 {
									fmt.Printf("Expected 1 output buffer, got %d", len(outputBuffers))
									t.Fail()
								}

								// Transfer output on-device buffer to a "host" value (in Go).
								output, err := pjrt.BufferToScalar[float32](outputBuffers[0])
								requireNoError(t, err, "Failed to transfer results of execution on input %g", input)

								// Print and check value is what we wanted.
								fmt.Printf("\tf(x=%g) = %g\n", input, output)
								assertInDelta(t, float64(wants[ii]), float64(output), 0.001)

								// Release inputBuffer and outputBuffers -- and don't wait for the GC.
								requireNoError(t, inputBuffer.Destroy())
								requireNoError(t, outputBuffers[0].Destroy())
							} // for range inputs
							// fmt.Printf("pjrt.BuffersAlive()=%d\n", pjrt.BuffersAlive())
						} // for range -e2e_repeat_execs
					} // for range devices (if portable)

					// We can now free the loadedExec (no need to wait for the GC).
					requireNoError(t, loadedExec.Destroy())
				} // for range -e2e_repeat_builds
			}) // t.Run(fixed/protable)
		} // fixed/portable
	}) // iterateClientsAndTest -- iterate clients.
}
