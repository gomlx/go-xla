package tests

import (
	"fmt"
	"strings"
	"testing"

	"github.com/gomlx/go-xla/pkg/pjrt"
	. "github.com/gomlx/go-xla/pkg/stablehlo"
	"github.com/gomlx/go-xla/pkg/types/dtypes"
	"github.com/gomlx/go-xla/pkg/types/shapes"
)

func TestCollectiveOps(t *testing.T) {
	iterateClientsAndTest(t, testCollectiveOps)
}

func testCollectiveOps(t *testing.T, client *pjrt.Client) {
	// We will test it with 2 devices.
	const numReplicas = 2
	numDevices := client.NumDevices()
	if numDevices < numReplicas {
		t.Skipf("Skipping test: not enough devices: %d < %d", numDevices, numReplicas)
		return
	}
	replicaGroups := [][]int{make([]int, numReplicas)}
	for i := range numReplicas {
		replicaGroups[0][i] = i
	}

	t.Run("CollectiveBroadcast", func(t *testing.T) {
		if strings.ToUpper(client.Plugin().Name()) == "CPU" {
			t.Skip("Skipping CollectiveBroadcast test: it is not implemented in PJRT CPU. ")
			return
		}
		b := New(t.Name()).WithNumReplicas(numReplicas)
		// SPMD program: takes one argument per replica.
		fn := b.Main()
		x := must1(fn.NamedInput("arg0", shapes.Make(dtypes.F32, 2)))
		// Broadcast %x (from replica 0) to all replicas.
		broadcasted := must1(CollectiveBroadcast(x, replicaGroups))
		must(fn.Return(broadcasted))
		program := must1(b.Build())
		fmt.Printf("%s program:\n%s\n", t.Name(), withLines(program))

		// Prepare inputs: one buffer for each replica.
		// Replica 0 has the data to be broadcasted.
		input0 := must1(client.BufferFromHost().FromFlatDataWithDimensions(
			[]float32{1.0, 2.0}, []int{2}).ToDeviceNum(replicaGroups[0][0]).Done())
		// Replica 1 has different data, which will be overwritten.
		input1 := must1(client.BufferFromHost().FromFlatDataWithDimensions(
			[]float32{7.0, 13.0}, []int{2}).ToDeviceNum(replicaGroups[0][1]).Done())

		// Execute expects a flat list of inputs, one for each argument of main(),
		// mapped to devices in order.
		e, err := client.Compile().WithStableHLO(program).WithSPMD(numReplicas).Done()
		if err != nil {
			t.Errorf("failed to compile program: \n%s\nError: %v", program, err)
			return
		}
		outputBuffers, err := e.Execute(input0, input1).DonateAll().Done()
		if err != nil {
			t.Errorf("failed to execute program: \n%s\nError: %v", program, err)
			return
		}

		// Check outputs: all replicas should have the data from replica 0.
		want := []FlatAndDims{
			{[]float32{1.0, 2.0}, []int{2}}, // Output on replica 0
			{[]float32{1.0, 2.0}, []int{2}}, // Output on replica 1
		}
		requireBuffersEqual(t, want, outputBuffers)
	})

	t.Run("AllReduce1", func(t *testing.T) {
		b := New(t.Name()).WithNumReplicas(numReplicas)

		// Define the main SPMD program.
		fn := b.Main()
		sumComputation := fn.Closure()
		{
			lhs := must1(sumComputation.NamedInput("lhs", shapes.Make(dtypes.F32)))
			rhs := must1(sumComputation.NamedInput("rhs", shapes.Make(dtypes.F32)))
			sum := must1(Add(lhs, rhs))
			must(sumComputation.Return(sum))
		}
		x0 := must1(fn.NamedInput("x", shapes.Make(dtypes.F32, 2))) // Input for replica 0
		reduced := must1(AllReduce([]*Value{x0}, replicaGroups, sumComputation))
		must(fn.Return(reduced[0]))
		program := must1(b.Build())
		fmt.Printf("%s program:\n%s", t.Name(), withLines(program))

		// Prepare inputs: one buffer for each replica.
		inputX0 := must1(client.BufferFromHost().FromFlatDataWithDimensions(
			[]float32{1.0, 10.0}, []int{2}).ToDeviceNum(replicaGroups[0][0]).Done())
		inputX1 := must1(client.BufferFromHost().FromFlatDataWithDimensions(
			[]float32{2.0, 20.0}, []int{2}).ToDeviceNum(replicaGroups[0][1]).Done())

		// Execute expects a flat list of inputs, one for each argument of main(),
		// mapped to devices in order.
		e, err := client.Compile().WithStableHLO(program).WithSPMD(numReplicas).Done()
		if err != nil {
			t.Errorf("failed to compile program: \n%s\nError: %v", program, err)
			return
		}
		outputBuffers, err := e.Execute(inputX0, inputX1).DonateAll().Done()
		if err != nil {
			t.Errorf("failed to execute program: \n%s\nError: %v", program, err)
			return
		}

		// Check outputs: all replicas should have the sum.
		want := []FlatAndDims{
			{[]float32{3.0, 30.0}, []int{2}}, // Output X on replica 0
			{[]float32{3.0, 30.0}, []int{2}}, // Output on replica 1
		}
		requireBuffersEqual(t, want, outputBuffers)
	})

	t.Run("AllReduce2", func(t *testing.T) {
		b := New(t.Name()).WithNumReplicas(numReplicas)

		// Define the main SPMD program.
		fn := b.Main()
		sumComputation := fn.Closure()
		{
			lhs := must1(sumComputation.NamedInput("lhs", shapes.Make(dtypes.F32)))
			rhs := must1(sumComputation.NamedInput("rhs", shapes.Make(dtypes.F32)))
			sum := must1(Add(lhs, rhs))
			must(sumComputation.Return(sum))
		}
		x := must1(fn.NamedInput("x", shapes.Make(dtypes.F32, 2))) // Input for replica 0
		y := must1(fn.NamedInput("y", shapes.Make(dtypes.F32, 3))) // Input for replica 0
		reduced := must1(AllReduce([]*Value{x, y}, replicaGroups, sumComputation))
		must(fn.Return(reduced[0], reduced[1]))
		program := must1(b.Build())
		fmt.Printf("%s program:\n%s", t.Name(), withLines(program))

		// Prepare inputs: one buffer for each replica.
		inputX0 := must1(client.BufferFromHost().FromFlatDataWithDimensions(
			[]float32{1.0, 10.0}, []int{2}).ToDeviceNum(replicaGroups[0][0]).Done())
		inputY0 := must1(client.BufferFromHost().FromFlatDataWithDimensions(
			[]float32{0.01, 0.1, 0.2}, []int{3}).ToDeviceNum(replicaGroups[0][0]).Done())
		inputX1 := must1(client.BufferFromHost().FromFlatDataWithDimensions(
			[]float32{2.0, 20.0}, []int{2}).ToDeviceNum(replicaGroups[0][1]).Done())
		inputY1 := must1(client.BufferFromHost().FromFlatDataWithDimensions(
			[]float32{0.05, 0.6, 0.7}, []int{3}).ToDeviceNum(replicaGroups[0][1]).Done())

		// Execute expects a flat list of inputs, one for each argument of main(),
		// mapped to devices in order.
		e, err := client.Compile().WithStableHLO(program).WithSPMD(numReplicas).Done()
		if err != nil {
			t.Errorf("failed to compile program: \n%s\nError: %v", program, err)
			return
		}
		outputBuffers, err := e.Execute(inputX0, inputY0, inputX1, inputY1).DonateAll().Done()
		if err != nil {
			t.Errorf("failed to execute program: \n%s\nError: %v", program, err)
			return
		}

		// Check outputs: all replicas should have the sum.
		want := []FlatAndDims{
			{[]float32{3.0, 30.0}, []int{2}},      // Output X on replica 0
			{[]float32{0.06, 0.7, 0.9}, []int{3}}, // Output Y on replica 0
			{[]float32{3.0, 30.0}, []int{2}},      // Output on replica 1
			{[]float32{0.06, 0.7, 0.9}, []int{3}}, // Output Y on replica 0
		}
		requireBuffersEqual(t, want, outputBuffers)
	})

	t.Run("AllGather", func(t *testing.T) {
		b := New(t.Name()).WithNumReplicas(numReplicas)
		fn := b.Main()
		x := must1(fn.NamedInput("x", shapes.Make(dtypes.F32, 2)))
		gathered := must1(AllGather(x, replicaGroups, 0))
		must(fn.Return(gathered))
		program := must1(b.Build())
		fmt.Printf("%s program:\n%s", t.Name(), withLines(program))

		input0 := must1(client.BufferFromHost().FromFlatDataWithDimensions(
			[]float32{1.0, 10.0}, []int{2}).ToDeviceNum(replicaGroups[0][0]).Done())
		input1 := must1(client.BufferFromHost().FromFlatDataWithDimensions(
			[]float32{2.0, 20.0}, []int{2}).ToDeviceNum(replicaGroups[0][1]).Done())

		e, err := client.Compile().WithStableHLO(program).WithSPMD(numReplicas).Done()
		if err != nil {
			t.Errorf("failed to compile program: \n%s\nError: %v", program, err)
			return
		}
		outputBuffers, err := e.Execute(input0, input1).DonateAll().Done()
		if err != nil {
			t.Errorf("failed to execute program: \n%s\nError: %v", program, err)
			return
		}

		want := []FlatAndDims{
			{[]float32{1.0, 10.0, 2.0, 20.0}, []int{4}},
			{[]float32{1.0, 10.0, 2.0, 20.0}, []int{4}},
		}
		requireBuffersEqual(t, want, outputBuffers)
	})

	t.Run("AllToAll", func(t *testing.T) {
		b := New(t.Name()).WithNumReplicas(numReplicas)
		fn := b.Main()
		x := must1(fn.NamedInput("x", shapes.Make(dtypes.F32, 4)))
		result := must1(AllToAll(x, replicaGroups, 0, 0, numReplicas))
		must(fn.Return(result))
		program := must1(b.Build())
		fmt.Printf("%s program:\n%s", t.Name(), withLines(program))

		input0 := must1(client.BufferFromHost().FromFlatDataWithDimensions(
			[]float32{1.0, 2.0, 3.0, 4.0}, []int{4}).ToDeviceNum(replicaGroups[0][0]).Done())
		input1 := must1(client.BufferFromHost().FromFlatDataWithDimensions(
			[]float32{10.0, 20.0, 30.0, 40.0}, []int{4}).ToDeviceNum(replicaGroups[0][1]).Done())

		e, err := client.Compile().WithStableHLO(program).WithSPMD(numReplicas).Done()
		if err != nil {
			t.Errorf("failed to compile program: \n%s\nError: %v", program, err)
			return
		}
		outputBuffers, err := e.Execute(input0, input1).DonateAll().Done()
		if err != nil {
			t.Errorf("failed to execute program: \n%s\nError: %v", program, err)
			return
		}

		want := []FlatAndDims{
			{[]float32{1.0, 2.0, 10.0, 20.0}, []int{4}},
			{[]float32{3.0, 4.0, 30.0, 40.0}, []int{4}},
		}
		requireBuffersEqual(t, want, outputBuffers)
	})

	t.Run("CollectivePermute", func(t *testing.T) {
		if strings.ToUpper(client.Plugin().Name()) == "CPU" {
			t.Skip("Skipping CollectivePermute test: it is not implemented in PJRT CPU. ")
			return
		}
		b := New(t.Name()).WithNumReplicas(numReplicas)
		fn := b.Main()
		x := must1(fn.NamedInput("x", shapes.Make(dtypes.F32, 2)))
		permuted := must1(CollectivePermute(x, [][2]int{{0, 1}, {1, 0}}))
		must(fn.Return(permuted))
		program := must1(b.Build())
		fmt.Printf("%s program:\n%s", t.Name(), withLines(program))

		input0 := must1(client.BufferFromHost().FromFlatDataWithDimensions(
			[]float32{1.0, 10.0}, []int{2}).ToDeviceNum(replicaGroups[0][0]).Done())
		input1 := must1(client.BufferFromHost().FromFlatDataWithDimensions(
			[]float32{2.0, 20.0}, []int{2}).ToDeviceNum(replicaGroups[0][1]).Done())

		e, err := client.Compile().WithStableHLO(program).WithSPMD(numReplicas).Done()
		if err != nil {
			t.Errorf("failed to compile program: \n%s\nError: %v", program, err)
			return
		}
		outputBuffers, err := e.Execute(input0, input1).DonateAll().Done()
		if err != nil {
			t.Errorf("failed to execute program: \n%s\nError: %v", program, err)
			return
		}

		want := []FlatAndDims{
			{[]float32{2.0, 20.0}, []int{2}},
			{[]float32{1.0, 10.0}, []int{2}},
		}
		requireBuffersEqual(t, want, outputBuffers)
	})
}
