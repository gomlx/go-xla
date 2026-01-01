package pjrt

import (
	"flag"
	"fmt"
	"runtime"
	"testing"
	"unsafe"

	"github.com/gomlx/go-xla/pkg/stablehlo"
	"github.com/gomlx/go-xla/pkg/types/dtypes"
	"github.com/gomlx/go-xla/pkg/types/shapes"
)

func TestScalarDataToRaw(t *testing.T) {
	rawData, dtype, dimensions := ScalarToRaw(uint32(3))
	assertEqual(t, dtype, dtypes.Uint32)
	assertEmpty(t, dimensions)
	assertEqual(t, uint32(3), *(*uint32)(unsafe.Pointer(unsafe.SliceData(rawData))))
}

func testTransfersImpl[T interface {
	float64 | float32 | int64 | int8
}](t *testing.T, client *Client) {
	var baseT T
	t.Run(fmt.Sprintf("%T", baseT), func(t *testing.T) {
		// Transfer arrays.
		input := []T{1, 2, 3}
		fmt.Printf("From %#v\n", input)
		buffer, err := ArrayToBuffer(client, input, 3, 1)
		requireNoError(t, err)
		assertFalse(t, buffer.IsShared())

		output, outputDims, err := BufferToArray[T](buffer)
		requireNoError(t, err)
		fmt.Printf("\t> output=%#v\n", output)
		assertEqualSlice(t, input, output)
		assertEqualSlice(t, []int{3, 1}, outputDims)

		flat, outputDims, err := buffer.ToFlatDataAndDimensions()
		requireNoError(t, err)
		assertEqualSlice(t, input, flat.([]T))
		assertEqualSlice(t, []int{3, 1}, outputDims)

		gotDevice, err := buffer.Device()
		requireNoError(t, err)
		wantDevice := client.AddressableDevices()[0]
		assertEqual(t, wantDevice.LocalHardwareID(), gotDevice.LocalHardwareID())
		assertEqual(t, 0, client.NumForDevice(gotDevice))

		// Try an invalid transfer: it should complain about the invalid dtype.
		_, _, err = BufferToArray[complex128](buffer)
		fmt.Printf("\t> expected wrong dtype error: %v\n", err)
		requireError(t, err)

		// Transfer scalars.
		from := T(13)
		fmt.Printf("From %T(%v)\n", from, from)
		buffer, err = ScalarToBuffer(client, from)
		requireNoError(t, err)
		to, err := BufferToScalar[T](buffer)
		requireNoError(t, err)
		fmt.Printf("\t> got %v\n", to)
		assertEqual(t, from, to)

		// ArrayToBuffer can also be used to transfer a scalar.
		from = T(19)
		fmt.Printf("From %T(%v)\n", from, from)
		buffer, err = ArrayToBuffer(client, []T{from})
		requireNoError(t, err)

		flatValues, dimensions, err := BufferToArray[T](buffer) // Check that it actually returns a scalar.
		requireNoError(t, err)
		assertLen(t, dimensions, 0) // That means, it is a scalar.
		fmt.Printf("\t> got %v\n", flatValues[0])
		assertEqual(t, from, flatValues[0])
	})
}

func TestTransfers(t *testing.T) {
	plugin, err := GetPlugin(*FlagPluginName)
	requireNoError(t, err)
	fmt.Printf("Loaded %s\n", plugin)

	client, err := plugin.NewClient(nil)
	requireNoError(t, err, "Failed to create a client on %s", plugin)
	fmt.Printf("%s\n", client)

	devices := client.AddressableDevices()
	assertNotEmpty(t, devices, "No addressable devices for client on %s", plugin)

	testTransfersImpl[float64](t, client)
	testTransfersImpl[float32](t, client)
	testTransfersImpl[int64](t, client)
	testTransfersImpl[int8](t, client)

	err = client.Destroy()
	requireNoError(t, err, "Failed to destroy client on %s", plugin)
}

func TestBufferProperties(t *testing.T) {
	plugin, err := GetPlugin(*FlagPluginName)
	requireNoError(t, err)
	fmt.Printf("Loaded %s\n", plugin)

	client, err := plugin.NewClient(nil)
	requireNoError(t, err, "Failed to create a client on %s", plugin)
	fmt.Printf("%s\n", client)

	{ // float32[3,4]
		dims := []int{3, 4}
		data := make([]float32, dims[0]*dims[1])

		buf, err := ArrayToBuffer(client, data, dims...)
		requireNoError(t, err)
		bufDims, err := buf.Dimensions()
		requireNoError(t, err)
		assertEqualSlice(t, dims, bufDims)
		dtype, err := buf.DType()
		requireNoError(t, err)
		assertEqual(t, dtypes.Float32, dtype)
	}

	{ // Scalar uint8
		buf, err := ScalarToBuffer(client, uint8(3))
		requireNoError(t, err)
		bufDims, err := buf.Dimensions()
		requireNoError(t, err)
		assertZero(t, len(bufDims))
		dtype, err := buf.DType()
		requireNoError(t, err)
		assertEqual(t, dtypes.Uint8, dtype)
	}
}

func TestBufferCopyToDevice(t *testing.T) {
	plugin, err := GetPlugin(*FlagPluginName)
	requireNoError(t, err)
	client, err := plugin.NewClient(nil)
	requireNoError(t, err, "Failed to create a client on %s", plugin)
	defer func() {
		err := client.Destroy()
		requireNoError(t, err)
	}()

	devices := client.AddressableDevices()
	if len(devices) < 2 {
		t.Skipf("TestBufferCopyToDevice requires at least 2 devices, only %d available", len(devices))
	}

	// Create a scalar buffer on the first device.
	device0 := devices[0]
	from := float32(42.0)
	bufferDev0, err := ScalarToBufferOnDeviceNum(client, 0, from)
	requireNoError(t, err)
	defer func() {
		err := bufferDev0.Destroy()
		requireNoError(t, err)
	}()

	// Verify it's on device 0
	bufferDevice, err := bufferDev0.Device()
	requireNoError(t, err)
	assertEqual(t, device0.LocalHardwareID(), bufferDevice.LocalHardwareID())

	// Copy buffer to the second device.
	device1 := devices[1]
	bufferDev1, err := bufferDev0.CopyToDevice(device1)
	requireNoError(t, err)
	defer func() {
		err := bufferDev1.Destroy()
		requireNoError(t, err)
	}()

	// Verify the new buffer is on device 1.
	bufferDevice, err = bufferDev1.Device()
	requireNoError(t, err)
	assertEqual(t, device1.LocalHardwareID(), bufferDevice.LocalHardwareID())

	// Verify the data is the same.
	to, err := BufferToScalar[float32](bufferDev1)
	requireNoError(t, err)
	assertEqual(t, from, to)
}

var flagForceSharedBuffer = flag.Bool(
	"force_shared_buffer", false, "Force executing TestCreateViewOfDeviceBuffer and TestBufferUnsafePointer even if plugin is not \"cpu\".")

func TestCreateViewOfDeviceBuffer(t *testing.T) {
	if *FlagPluginName != "cpu" && !*flagForceSharedBuffer {
		t.Skip("Skipping TestCreateViewOfDeviceBuffer because -plugin != \"cpu\". " +
			"Set --force_create_view to force executing the test anyway")
	}

	// Create plugin.
	plugin := must1(GetPlugin(*FlagPluginName))
	client := must1(plugin.NewClient(nil))
	defer runtime.KeepAlive(client)

	// f(x) = x + 1
	dtype := dtypes.Float32
	shape := shapes.Make(dtype, 2, 3)
	builder := stablehlo.New("Add1")
	mainFn := builder.Main()
	x := must1(mainFn.NamedInput("x", shape))
	one := must1(mainFn.ConstantFromScalar(float32(1)))
	broadcastedOne := must1(stablehlo.BroadcastInDim(one, x.Shape(), nil))
	add1 := must1(stablehlo.Add(x, broadcastedOne))
	must(mainFn.Return(add1))
	compBytes := must1(builder.Build())
	exec := must1(client.Compile().WithStableHLO(compBytes).Done())

	// Input is created as a "Device Buffer View"
	storage := AlignedAlloc(shape.Memory(), BufferAlignment)
	defer AlignedFree(storage)
	inputBuffer, err := client.CreateViewOfDeviceBuffer(storage, dtype, shape.Dimensions)
	requireNoError(t, err)
	defer func() {
		err := inputBuffer.Destroy()
		if err != nil {
			t.Logf("Failed Buffer.Destroy(): %+v", err)
		}
	}()

	flatData := unsafe.Slice((*float32)(storage), shape.Size())
	for ii := range flatData {
		flatData[ii] = float32(ii)
	}
	assertTrue(t, inputBuffer.IsShared())

	results, err := exec.Execute(inputBuffer).DonateNone().Done()
	requireNoError(t, err)
	assertLen(t, results, 1)

	gotFlat, gotDims, err := BufferToArray[float32](results[0])
	requireNoError(t, err)
	assertEqualSlice(t, shape.Dimensions, gotDims)
	assertEqualSlice(t, []float32{1, 2, 3, 4, 5, 6}, gotFlat)

	// Change the buffer directly, and see that we can reuse the buffer in PJRT, without the extra transfer.
	flatData[1] = 11
	results, err = exec.Execute(inputBuffer).DonateNone().Done()
	requireNoError(t, err)
	assertLen(t, results, 1)
	gotFlat, gotDims, err = BufferToArray[float32](results[0])
	requireNoError(t, err)
	assertEqualSlice(t, shape.Dimensions, gotDims)
	assertEqualSlice(t, []float32{1, 12, 3, 4, 5, 6}, gotFlat)
	requireNoError(t, inputBuffer.Destroy())
}

func TestNewSharedBuffer(t *testing.T) {
	if *FlagPluginName != "cpu" && !*flagForceSharedBuffer {
		t.Skip("Skipping TestNewSharedBuffer because -plugin != \"cpu\". " +
			"Set --force_create_view to force executing the test anyway")
	}

	// Create plugin.
	plugin := must1(GetPlugin(*FlagPluginName))
	client := must1(plugin.NewClient(nil))
	defer runtime.KeepAlive(client)

	// f(x) = x + 1
	dtype := dtypes.Float32
	shape := shapes.Make(dtype, 2, 3)
	builder := stablehlo.New("Add1")
	mainFn := builder.Main()
	x := must1(mainFn.NamedInput("x", shape))
	one := must1(mainFn.ConstantFromScalar(float32(1)))
	broadcastedOne := must1(stablehlo.BroadcastInDim(one, x.Shape(), nil))
	add1 := must1(stablehlo.Add(x, broadcastedOne))
	must(mainFn.Return(add1))
	compBytes := must1(builder.Build())
	exec := must1(client.Compile().WithStableHLO(compBytes).Done())

	// Input is created as a "Device Buffer View"
	inputBuffer, flatAny, err := client.NewSharedBuffer(dtype, shape.Dimensions)
	requireNoError(t, err)
	defer func() {
		err := inputBuffer.Destroy()
		if err != nil {
			t.Logf("Failed to destroy shared buffer: %+v", err)
		}
	}()

	flatData := flatAny.([]float32)
	for ii := range flatData {
		flatData[ii] = float32(ii)
	}
	assertTrue(t, inputBuffer.IsShared())

	results, err := exec.Execute(inputBuffer).DonateNone().Done()
	requireNoError(t, err)
	assertLen(t, results, 1)

	gotFlat, gotDims, err := BufferToArray[float32](results[0])
	requireNoError(t, err)
	assertEqualSlice(t, shape.Dimensions, gotDims)
	assertEqualSlice(t, []float32{1, 2, 3, 4, 5, 6}, gotFlat)

	// Change the buffer directly, and see that we can reuse the buffer in PJRT, without the extra transfer.
	flatData[1] = 11
	results, err = exec.Execute(inputBuffer).DonateNone().Done()
	requireNoError(t, err)
	assertLen(t, results, 1)
	gotFlat, gotDims, err = BufferToArray[float32](results[0])
	requireNoError(t, err)
	assertEqualSlice(t, shape.Dimensions, gotDims)
	assertEqualSlice(t, []float32{1, 12, 3, 4, 5, 6}, gotFlat)

	requireNoError(t, inputBuffer.Destroy())
}

func TestBufferData(t *testing.T) {
	if *FlagPluginName != "cpu" && !*flagForceSharedBuffer {
		t.Skip("Skipping TestNewSharedBuffer because -plugin != \"cpu\". " +
			"Set --force_create_view to force executing the test anyway")
	}

	// Create plugin.
	plugin := must1(GetPlugin(*FlagPluginName))
	client := must1(plugin.NewClient(nil))
	defer runtime.KeepAlive(client)

	// f(x) = x + 1
	dtype := dtypes.Float32
	shape := shapes.Make(dtype, 2, 3)
	builder := stablehlo.New("Add1")
	mainFn := builder.Main()
	x := must1(mainFn.NamedInput("x", shape))
	one := must1(mainFn.ConstantFromScalar(float32(1)))
	broadcastedOne := must1(stablehlo.BroadcastInDim(one, x.Shape(), nil))
	add1 := must1(stablehlo.Add(x, broadcastedOne))
	must(mainFn.Return(add1))
	compBytes := must1(builder.Build())
	exec := must1(client.Compile().WithStableHLO(compBytes).Done())

	// Input is created as a "Device Buffer View"
	inputBuffer, flatAny, err := client.NewSharedBuffer(dtype, shape.Dimensions)
	requireNoError(t, err)
	defer func() {
		err := inputBuffer.Destroy()
		if err != nil {
			t.Logf("Failed to destroy shared buffer: %+v", err)
		}
	}()

	flatData := flatAny.([]float32)
	for ii := range flatData {
		flatData[ii] = float32(ii)
	}
	assertTrue(t, inputBuffer.IsShared())

	results, err := exec.Execute(inputBuffer).DonateNone().Done()
	requireNoError(t, err)
	assertLen(t, results, 1)

	flatOutput, err := results[0].Data()
	requireNoError(t, err)
	assertEqualSlice(t, []float32{1, 2, 3, 4, 5, 6}, flatOutput.([]float32))
}

func TestBufferDestroyAfterClient(t *testing.T) {
	// Create the plugin and the client.
	plugin := must1(GetPlugin(*FlagPluginName))
	client := must1(plugin.NewClient(nil))
	defer runtime.KeepAlive(client)

	// Create buffer
	buffer1, err := ScalarToBuffer(client, float32(7))
	requireNoError(t, err)
	buffer2, err := ScalarToBuffer(client, float32(42))
	requireNoError(t, err)

	// Destroy buffer1 before the client is destroyed.
	requireNotPanics(t, func() {
		err = buffer1.Destroy()
	})

	// Destroy the client.
	err = client.Destroy()
	requireNoError(t, err)

	// Destroy buffer2 after the client is destroyed: it should be safe!
	requireNotPanics(t, func() {
		err = buffer2.Destroy()
	})
	requireNoError(t, err)
}
