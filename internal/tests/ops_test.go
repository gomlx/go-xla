package tests

import (
	"flag"
	"fmt"
	"iter"
	"math"
	"math/bits"
	"reflect"
	"strings"
	"testing"

	"github.com/gomlx/go-xla/pkg/pjrt"
	"github.com/gomlx/go-xla/pkg/stablehlo"
	. "github.com/gomlx/go-xla/pkg/stablehlo"
	"github.com/gomlx/go-xla/pkg/types"
	"github.com/gomlx/go-xla/pkg/types/dtypes"
	"github.com/gomlx/go-xla/pkg/types/shapes"
	"k8s.io/klog/v2"
)

var flagPluginNames = flag.String("plugins", "cpu", "List (|-separated) of PJRT plugin names or full paths. E.g. \"cpu|cuda\"")

func must(err error) {
	if err != nil {
		klog.Errorf("%+v", err)
		panic(err)
	}
}

func must1[T any](value T, err error) T {
	must(err)
	return value
}

func must2[T1, T2 any](value1 T1, value2 T2, err error) (T1, T2) {
	must(err)
	return value1, value2
}

// withLines prefix each line of text with a "%04d: " of the line number.
func withLines(text []byte) string {
	var result strings.Builder
	lines := strings.Split(string(text), "\n")
	for i, line := range lines {
		_, _ = fmt.Fprintf(&result, "%04d: %s\n", i+1, line)
	}
	return result.String()
}

func getPluginNames() []string {
	names := strings.Split(*flagPluginNames, "|")
	var to int
	for _, name := range names {
		if name != "" {
			names[to] = name
			to++
		}
	}
	if to == 0 {
		panic("no XLA plugin names defined with -plugins")
	}
	names = names[:to]
	return names
}

func pjrtNumClients() int {
	return len(getPluginNames())
}

func pjrtClientsIterator(t *testing.T) iter.Seq2[string, *pjrt.Client] {
	return func(yield func(string, *pjrt.Client) bool) {
		pluginNames := getPluginNames()
		fmt.Printf("Plugins: %v\n", pluginNames)
		for _, pluginName := range pluginNames {
			plugin, err := pjrt.GetPlugin(pluginName)
			if err != nil {
				t.Fatalf("failed to load plugin %q: %v", pluginName, err)
			}
			fmt.Printf("Plugin: %s", plugin)
			client, err := plugin.NewClient(nil)
			if err != nil {
				t.Fatalf("failed to create client for plugin %q: %v", pluginName, err)
			}
			fmt.Printf("- Client %s (version %s): %d devices\n",
				client.Platform(), client.PlatformVersion(), client.NumDevices())
			ok := yield(pluginName, client)
			if err := client.Destroy(); err != nil {
				t.Fatalf("failed to destroy client: %v", err)
			}
			if !ok {
				return
			}
		}
	}
}

func iterateClientsAndTest(t *testing.T, testFn func(*testing.T, *pjrt.Client)) {
	numClients := pjrtNumClients()
	for pluginName, client := range pjrtClientsIterator(t) {
		if numClients > 1 {
			t.Run(pluginName, func(t *testing.T) {
				testFn(t, client)
			})
		} else {
			testFn(t, client)
		}
		err := client.Destroy()
		if err != nil {
			t.Fatalf("failed to destroy the client %q", pluginName)
		}
	}
}

// compileAndExecute program with PJRT. All inputs are donated.
func compileAndExecute(t *testing.T, client *pjrt.Client, program []byte, inputs ...*pjrt.Buffer) []*pjrt.Buffer {
	loadedExec, err := client.Compile().WithStableHLO(program).Done()
	if err != nil {
		t.Fatalf("failed to compile program: \n%s\nError: %v", program, err)
	}
	defer func() {
		err := loadedExec.Destroy()
		if err != nil {
			t.Errorf("failed to destroy loaded exec: %+v", err)
		}
	}()
	outputBuffers, err := loadedExec.Execute(inputs...).DonateAll().Done()
	if err != nil {
		t.Fatalf("failed to execute program: \n%s\nError: %v", program, err)
	}
	return outputBuffers
}

type FlatAndDims struct {
	Flat any
	Dims []int
}

// requireBuffersEqual checks that the actual buffers contents match the expected flat values.
// It destroys the buffers.
func requireBuffersEqual(t *testing.T, expected []FlatAndDims, got []*pjrt.Buffer) {
	defer func() {
		for _, b := range got {
			err := b.Destroy()
			if err != nil {
				t.Errorf("failed to destroy buffer: %+v", err)
			}
		}
	}()
	if len(got) != len(expected) {
		t.Fatalf("expected %d outputs, got %d", len(expected), len(got))
	}
	for i, b := range got {
		gotFlat, gotDims, err := b.ToFlatDataAndDimensions()
		if err != nil {
			t.Fatalf("failed to get buffer contents for output #%d: %v", i, err)
		}
		expectedShape, err := shapes.FromAnyValue(expected[i].Flat)
		if err != nil {
			t.Fatalf("failed to get shape for output #%d: %v\nValue: %v", i, err, expected[i].Flat)
		}
		dtype := expectedShape.DType
		fmt.Printf("\t - output #%d:\n\t   - Got: dims=%v, flat_values=%v\n", i, gotDims, gotFlat)
		fmt.Printf("\t   - Want(%s): dims=%v, flat_values=%v\n", dtype, expected[i].Dims, expected[i].Flat)

		if !reflect.DeepEqual(expected[i].Dims, gotDims) {
			// Handle nil vs empty slice difference.
			if len(expected[i].Dims) != 0 || len(gotDims) != 0 {
				t.Errorf("output #%d dims don't match: want %v, got %v", i, expected[i].Dims, gotDims)
			}
		}

		switch dtype {
		case dtypes.Float64, dtypes.Float32:
			// For floats use InDelta-like comparison.
			expVal := reflect.ValueOf(expected[i].Flat)
			gotVal := reflect.ValueOf(gotFlat)
			if expVal.Len() != gotVal.Len() {
				t.Errorf("output #%d flat values length mismatch: want %d, got %d", i, expVal.Len(), gotVal.Len())
				continue
			}
			for j := 0; j < expVal.Len(); j++ {
				e := expVal.Index(j).Float()
				g := gotVal.Index(j).Float()
				diff := math.Abs(e - g)
				if diff > 1e-4 {
					t.Errorf("output #%d flat values don't match at index %d: want %v, got %v (diff %v)", i, j, e, g, diff)
					break // Stop after first error to avoid spam
				}
			}
		default:
			if !reflect.DeepEqual(expected[i].Flat, gotFlat) {
				t.Errorf("output #%d flat values don't match: want %v, got %v", i, expected[i].Flat, gotFlat)
			}
		}
	}
}

// TestRendering checks for special cases of the StableHLO parser.
func TestRendering(t *testing.T) {
	iterateClientsAndTest(t, testRendering)
}

func testRendering(t *testing.T, client *pjrt.Client) {
	t.Run("Return-multi-output", func(t *testing.T) {
		b := New(t.Name())
		fn := b.NewFunction("main")
		c1 := must1(fn.ConstantFromScalar(1.0))
		c2 := must1(fn.ConstantFromScalar(2.0))
		c3 := must1(fn.ConstantFromScalar(float32(math.Inf(-1))))
		c4 := must1(fn.ConstantFromScalar(math.Inf(1)))
		sum := must1(Add(c1, c2))
		must(fn.Return(c1, c2, sum, c3, c4))
		program := must1(b.Build())
		fmt.Printf("%s program:\n%s", t.Name(), withLines(program))
		output := compileAndExecute(t, client, program)
		requireBuffersEqual(t, []FlatAndDims{
			{[]float64{1}, nil},
			{[]float64{2}, nil},
			{[]float64{3}, nil},
			{[]float32{float32(math.Inf(-1))}, nil},
			{[]float64{math.Inf(1)}, nil},
		}, output)
	})

	// Floats checks that different floats are properly rendered -- StableHLO is very particular on how it parses.
	t.Run("Floats", func(t *testing.T) {
		builder := New(t.Name())
		fn := builder.Main()
		x0 := must1(fn.ConstantFromFlatAndDimensions([]float64{1e6, 1e-6, 0, -1e-8, -1e6, 1.2345678923456e78}, 6))
		x1 := must1(fn.ConstantFromFlatAndDimensions([]float64{1e6, 1e-6, 0, -1e-8, -1e6}, 5, 1))
		must(fn.Return(x0, x1))
		program := must1(builder.Build())
		fmt.Printf("%s program:\n%s", t.Name(), withLines(program))
		outputs := compileAndExecute(t, client, program)
		requireBuffersEqual(t, []FlatAndDims{
			{[]float64{1e6, 1e-6, 0, -1e-8, -1e6, 1.2345678923456e78}, []int{6}},
			{[]float64{1e6, 1e-6, 0, -1e-8, -1e6}, []int{5, 1}},
		}, outputs)
	})
}

func TestOps(t *testing.T) {
	iterateClientsAndTest(t, testOps)
}

func testOps(t *testing.T, client *pjrt.Client) {
	t.Run("Complex", func(t *testing.T) {
		builder := New(t.Name())
		fn := builder.Main()
		shape := shapes.Make(dtypes.Float64)
		lhsV, rhsV := must1(fn.NamedInput("lhs", shape)), must1(fn.NamedInput("rhs", shape))
		must(fn.Return(must1(Complex(lhsV, rhsV))))
		program := must1(builder.Build())
		fmt.Printf("%s program:\n%s", t.Name(), withLines(program))
		a := must1(client.BufferFromHost().FromFlatDataWithDimensions([]float64{1.0}, nil).Done())
		b := must1(client.BufferFromHost().FromFlatDataWithDimensions([]float64{-1.0}, nil).Done())
		output := compileAndExecute(t, client, program, a, b)
		requireBuffersEqual(t, []FlatAndDims{{[]complex128{1 - 1i}, nil}}, output)
	})

	t.Run("Clamp", func(t *testing.T) {
		builder := New(t.Name())
		fn := builder.Main()
		minV := must1(fn.NamedInput("min", shapes.Make(dtypes.Float32)))
		xV := must1(fn.NamedInput("x", shapes.Make(dtypes.Float32, 3)))
		maxV := must1(fn.NamedInput("max", shapes.Make(dtypes.Float32)))
		must(fn.Return(must1(Clamp(minV, xV, maxV))))
		program := must1(builder.Build())
		fmt.Printf("%s program:\n%s", t.Name(), withLines(program))
		minArg := must1(client.BufferFromHost().FromFlatDataWithDimensions([]float32{-1.0}, nil).Done())
		maxArg := must1(client.BufferFromHost().FromFlatDataWithDimensions([]float32{1.0}, nil).Done())
		x := must1(client.BufferFromHost().FromFlatDataWithDimensions([]float32{0.1, -2.2, 3.3}, []int{3}).Done())
		output := compileAndExecute(t, client, program, minArg, x, maxArg)
		requireBuffersEqual(t, []FlatAndDims{{[]float32{0.1, -1, 1}, []int{3}}}, output)
	})

	t.Run("Iota", func(t *testing.T) {
		builder := New(t.Name())
		fn := builder.Main()
		must(fn.Return(
			must1(fn.Iota(shapes.Make(dtypes.F32, 2, 2), 0)),
			must1(fn.Iota(shapes.Make(dtypes.F32, 2, 2), 1)),
			must1(fn.Iota(shapes.Make(dtypes.F32, 4), 0)),
		))
		program := must1(builder.Build())
		fmt.Printf("%s program:\n%s", t.Name(), withLines(program))
		outputs := compileAndExecute(t, client, program)
		requireBuffersEqual(t, []FlatAndDims{
			{[]float32{0, 0, 1, 1}, []int{2, 2}},
			{[]float32{0, 1, 0, 1}, []int{2, 2}},
			{[]float32{0, 1, 2, 3}, []int{4}},
		}, outputs)
	})

	t.Run("IsFinite", func(t *testing.T) {
		builder := New(t.Name())
		fn := builder.Main()
		input := must1(fn.NamedInput("x", shapes.Make(dtypes.F64, 6)))
		must(fn.Return(must1(IsFinite(input))))
		program := must1(builder.Build())
		fmt.Printf("%s program:\n%s", t.Name(), withLines(program))
		v := must1(client.BufferFromHost().FromFlatDataWithDimensions([]float64{0, -1, 1, math.Inf(1), math.Inf(-1), math.NaN()}, []int{6}).Done())
		outputs := compileAndExecute(t, client, program, v)
		requireBuffersEqual(t, []FlatAndDims{
			{[]bool{true, true, true, false, false, false}, []int{6}},
		}, outputs)
	})

	t.Run("Reshape", func(t *testing.T) {
		builder := New(t.Name())
		fn := builder.Main()
		x := must1(fn.Iota(shapes.Make(dtypes.F32, 3, 2), 0))
		y := must1(Reshape(x, shapes.Make(dtypes.F32, 2, 3)))
		must(fn.Return(y))
		program := must1(builder.Build())
		fmt.Printf("%s program:\n%s", t.Name(), withLines(program))
		outputs := compileAndExecute(t, client, program)
		requireBuffersEqual(t, []FlatAndDims{
			{[]float32{0, 0, 1, 1, 2, 2}, []int{2, 3}},
		}, outputs)
	})

	t.Run("BroadcastInDim", func(t *testing.T) {
		builder := New(t.Name())
		fn := builder.Main()
		x := must1(fn.Iota(shapes.Make(dtypes.F32, 3), 0))
		y := must1(BroadcastInDim(x, shapes.Make(dtypes.F32, 2, 3), []int{1}))
		must(fn.Return(y))
		program := must1(builder.Build())
		fmt.Printf("%s program:\n%s", t.Name(), withLines(program))
		outputs := compileAndExecute(t, client, program)
		requireBuffersEqual(t, []FlatAndDims{
			{[]float32{0, 1, 2, 0, 1, 2}, []int{2, 3}},
		}, outputs)
	})

	t.Run("BroadcastInDim<scalar>", func(t *testing.T) {
		builder := New(t.Name())
		fn := builder.Main()
		x := must1(fn.ConstantFromScalar(float32(7.0)))
		y := must1(BroadcastInDim(x, shapes.Make(dtypes.F32, 3), nil))
		must(fn.Return(y))
		program := must1(builder.Build())
		fmt.Printf("%s program:\n%s", t.Name(), withLines(program))
		outputs := compileAndExecute(t, client, program)
		requireBuffersEqual(t, []FlatAndDims{
			{[]float32{7, 7, 7}, []int{3}},
		}, outputs)
	})

	t.Run("Gather", func(t *testing.T) {
		builder := New(t.Name())
		fn := builder.Main()
		x := must1(fn.Iota(shapes.Make(dtypes.F32, 3*5), 0))
		x = must1(Reshape(x, shapes.Make(dtypes.F32, 3, 5)))
		indices := must1(fn.ConstantFromFlatAndDimensions([]int{2, 0}, 2, 1))
		offsetOutputAxes := []int{1}
		collapsedSliceAxes := []int{0}
		var operandBatchingAxes, startIndicesBatchingAxes []int
		startIndexMap := []int{0}
		sliceSizes := []int{1, 5}
		y := must1(Gather(x, indices, 1,
			offsetOutputAxes, collapsedSliceAxes, operandBatchingAxes,
			startIndicesBatchingAxes, startIndexMap,
			sliceSizes, false))
		must(fn.Return(y))
		program := must1(builder.Build())
		fmt.Printf("%s program:\n%s", t.Name(), withLines(program))
		outputs := compileAndExecute(t, client, program)
		requireBuffersEqual(t, []FlatAndDims{
			{[]float32{ /* row=0: */ 10, 11, 12, 13, 14 /* row=1: */, 0, 1, 2, 3, 4}, []int{2, 5}},
		}, outputs)
	})

	//	Slice(x={0, 1, 2, 3, 4}, starts={2}, limits={4}, strides=nil) -> {2, 3}
	//	Slice(x={0, 1, 2, 3, 4}, starts={2}, limits={5}, strides={2}) -> {2, 4}
	t.Run("Slice", func(t *testing.T) {
		builder := New(t.Name())
		fn := builder.Main()
		x := must1(fn.Iota(shapes.Make(dtypes.F32, 5), 0))
		y0 := must1(Slice(x, []int{2}, []int{4}, nil))
		y1 := must1(Slice(x, []int{2}, []int{5}, []int{2}))
		must(fn.Return(y0, y1))
		program := must1(builder.Build())
		fmt.Printf("%s program:\n%s", t.Name(), withLines(program))
		outputs := compileAndExecute(t, client, program)
		requireBuffersEqual(t, []FlatAndDims{
			{[]float32{2, 3}, []int{2}},
			{[]float32{2, 4}, []int{2}},
		}, outputs)
	})

	t.Run("Concatenate", func(t *testing.T) {
		builder := New(t.Name())
		fn := builder.Main()
		x := must1(fn.Iota(shapes.Make(dtypes.F32, 2, 3), 1))
		y := must1(fn.Iota(shapes.Make(dtypes.F32, 2, 1), 0))
		z := must1(Concatenate(1, x, y))
		must(fn.Return(z))
		program := must1(builder.Build())
		fmt.Printf("%s program:\n%s", t.Name(), withLines(program))
		outputs := compileAndExecute(t, client, program)
		requireBuffersEqual(t, []FlatAndDims{
			{[]float32{0, 1, 2, 0, 0, 1, 2, 1}, []int{2, 4}},
		}, outputs)
	})

	t.Run("Reduce", func(t *testing.T) {
		builder := New(t.Name())
		fn := builder.Main()
		x := must1(fn.Iota(shapes.Make(dtypes.F32, 2*3), 0))
		x = must1(Reshape(x, shapes.Make(dtypes.F32, 2, 3)))
		zero := must1(fn.ConstantFromScalar(float32(0)))
		reductionFn := fn.Closure()
		lhs := must1(reductionFn.NamedInput("lhs", shapes.Make(dtypes.F32)))
		rhs := must1(reductionFn.NamedInput("rhs", shapes.Make(dtypes.F32)))
		must(reductionFn.Return(must1(Add(lhs, rhs))))
		r0 := must1(Reduce(x, zero, reductionFn, 1))
		r1 := must1(Reduce(x, zero, reductionFn, 0))
		must(fn.Return(r0, r1))
		program := must1(builder.Build())
		fmt.Printf("%s program:\n%s", t.Name(), withLines(program))
		outputs := compileAndExecute(t, client, program)
		requireBuffersEqual(t, []FlatAndDims{
			{[]float32{3, 12}, []int{2}},
			{[]float32{3, 5, 7}, []int{3}},
		}, outputs)
	})

	t.Run("MultiReduce", func(t *testing.T) {
		builder := New(t.Name())
		fn := builder.Main()
		x := must1(fn.Iota(shapes.Make(dtypes.F32, 2*3), 0))
		x = must1(Reshape(x, shapes.Make(dtypes.F32, 2, 3)))
		y := must1(fn.Iota(shapes.Make(dtypes.Int32, 2*3), 0))
		y = must1(Reshape(y, shapes.Make(dtypes.Int32, 2, 3)))
		zeroF32 := must1(fn.ConstantFromScalar(float32(0)))
		zeroI32 := must1(fn.ConstantFromScalar(int32(0)))
		reductionFn := fn.Closure()
		lhs0 := must1(reductionFn.NamedInput("lhs0", shapes.Make(dtypes.F32)))
		lhs1 := must1(reductionFn.NamedInput("lhs1", shapes.Make(dtypes.Int32)))
		rhs0 := must1(reductionFn.NamedInput("rhs0", shapes.Make(dtypes.F32)))
		rhs1 := must1(reductionFn.NamedInput("rhs1", shapes.Make(dtypes.Int32)))
		must(reductionFn.Return(
			must1(Add(lhs0, rhs0)),
			must1(Add(lhs1, rhs1))))
		results := must1(MultiReduce(
			[]*Value{x, y},
			[]*Value{zeroF32, zeroI32}, reductionFn, 1))
		must(fn.Return(results[0], results[1]))
		program := must1(builder.Build())
		fmt.Printf("%s program:\n%s", t.Name(), withLines(program))
		outputs := compileAndExecute(t, client, program)
		requireBuffersEqual(t, []FlatAndDims{
			{[]float32{3, 12}, []int{2}},
			{[]int32{3, 12}, []int{2}},
		}, outputs)
	})

	t.Run("Select", func(t *testing.T) {
		builder := New(t.Name())
		fn := builder.Main()
		pred0 := must1(fn.ConstantFromFlatAndDimensions([]bool{true, false, true}, 3))
		pred1 := must1(fn.ConstantFromScalar(false))
		onTrue := must1(fn.Iota(shapes.Make(dtypes.F32, 3), 0))
		onFalse := must1(Negate(onTrue))
		result0 := must1(Select(pred0, onTrue, onFalse))
		result1 := must1(Select(pred1, onTrue, onFalse))
		must(fn.Return(result0, result1))
		program := must1(builder.Build())
		fmt.Printf("%s program:\n%s", t.Name(), withLines(program))
		outputs := compileAndExecute(t, client, program)
		requireBuffersEqual(t, []FlatAndDims{
			{[]float32{0, -1, 2}, []int{3}},
			{[]float32{0, -1, -2}, []int{3}},
		}, outputs)
	})

	t.Run("BitcastConvert", func(t *testing.T) {
		builder := New(t.Name())
		fn := builder.Main()
		c0 := must1(fn.ConstantFromFlatAndDimensions([]uint16{0xbeef, 0xdead}, 1, 2))
		c1 := must1(fn.ConstantFromScalar(uint32(0xdeadbeef)))
		c2 := must1(fn.ConstantFromScalar(uint32(0x7F800000)))
		must(fn.Return(
			must1(BitcastConvert(c0, dtypes.Uint32)),
			must1(BitcastConvert(c1, dtypes.Uint16)),
			must1(BitcastConvert(c2, dtypes.F32)),
		))
		program := must1(builder.Build())
		fmt.Printf("%s program:\n%s", t.Name(), withLines(program))
		outputs := compileAndExecute(t, client, program)
		requireBuffersEqual(t, []FlatAndDims{
			{[]uint32{0xdeadbeef}, []int{1}},
			{[]uint16{0xbeef, 0xdead}, []int{2}},
			{[]float32{float32(math.Inf(1))}, nil},
		}, outputs)
	})

	t.Run("Transpose", func(t *testing.T) {
		builder := New(t.Name())
		fn := builder.Main()
		x := must1(fn.Iota(shapes.Make(dtypes.F32, 2*3), 0))
		x = must1(Reshape(x, shapes.Make(dtypes.F32, 2, 3)))
		must(fn.Return(must1(Transpose(x, 1, 0))))
		program := must1(builder.Build())
		fmt.Printf("%s program:\n%s", t.Name(), withLines(program))
		outputs := compileAndExecute(t, client, program)
		requireBuffersEqual(t, []FlatAndDims{
			{[]float32{0, 3, 1, 4, 2, 5}, []int{3, 2}},
		}, outputs)
	})

	for _, algo := range []types.RNGBitGeneratorAlgorithm{types.RNGDefault, types.RNGThreeFry, types.RNGPhilox} {
		t.Run(fmt.Sprintf("RNGBitGenerator-%s", algo), func(t *testing.T) {
			builder := New(t.Name())
			fn := builder.Main()
			state := must1(fn.ConstantFromFlatAndDimensions([]uint64{42, 1}, 2))
			const numSamples = 10_000
			_, noiseV := must2(RNGBitGenerator(state, shapes.Make(dtypes.Uint64, numSamples), algo))
			must(fn.Return(noiseV))
			program := must1(builder.Build())
			fmt.Printf("%s program:\n%s", t.Name(), withLines(program))
			outputs := compileAndExecute(t, client, program)
			flat, dims, err := outputs[0].ToFlatDataAndDimensions()
			if err != nil {
				t.Fatalf("ToFlatDataAndDimensions error: %v", err)
			}
			if !reflect.DeepEqual([]int{numSamples}, dims) {
				t.Errorf("dims mismatch: want [%d], got %v", numSamples, dims)
			}
			noise := flat.([]uint64)
			// Count bits in each uint64
			var totalBits int
			for _, n := range noise {
				totalBits += bits.OnesCount64(n)
			}
			// We expect roughly 32 bits per number +/- 2 standard deviations
			expectedBits := 32 * numSamples
			fmt.Printf("\tgot %d bits set, expected %d\n", totalBits, expectedBits)
			margin := 2 * numSamples
			if totalBits <= expectedBits-margin {
				t.Errorf("totalBits %d <= expectedBits-margin %d", totalBits, expectedBits-margin)
			}
			if totalBits >= expectedBits+margin {
				t.Errorf("totalBits %d >= expectedBits+margin %d", totalBits, expectedBits+margin)
			}
		})
	}

	t.Run("Scatter", func(t *testing.T) {
		builder := New(t.Name())
		fn := builder.Main()
		zeroF32 := must1(fn.ConstantFromScalar(float32(0)))
		zeroI32 := must1(fn.ConstantFromScalar(int32(0)))
		inputsF32 := must1(BroadcastInDim(zeroF32, shapes.Make(dtypes.F32, 2, 1, 2, 3), nil))
		inputsI32 := must1(BroadcastInDim(zeroI32, shapes.Make(dtypes.Int32, 2, 1, 2, 3), nil))
		scatterIndices := must1(fn.ConstantFromFlatAndDimensions(
			[]int32{1, 1, 0, 0, 0, 2, 0, 0}, 2, 2, 2))

		updatesF32 := must1(fn.Iota(shapes.Make(dtypes.F32, 2*2*1), 0))
		updatesF32 = must1(Reshape(updatesF32, shapes.Make(dtypes.F32, 2, 2, 1)))
		hundredF32 := must1(BroadcastInDim(
			must1(fn.ConstantFromScalar(float32(100))), updatesF32.Shape(), nil))
		updatesF32 = must1(Add(updatesF32, hundredF32))

		updatesI32 := must1(fn.Iota(shapes.Make(dtypes.Int32, 2*2*1), 0))
		updatesI32 = must1(Reshape(updatesI32, shapes.Make(dtypes.Int32, 2, 2, 1)))
		thousandI32 := must1(BroadcastInDim(
			must1(fn.ConstantFromScalar(int32(1000))), updatesI32.Shape(), nil))
		updatesI32 = must1(Add(updatesI32, thousandI32))

		updateWindowAxes := []int{2}
		insertedWindowAxes := []int{1, 2}
		inputBatchingAxes := []int{0}
		scatterIndicesBatchingAxes := []int{0}
		indexedInputAxes := []int{2, 3}
		indexVectorAxis := 2
		updateFn := fn.Closure()
		lhsF32 := must1(updateFn.NamedInput("lhsF32", shapes.Make(dtypes.F32)))
		lhsI32 := must1(updateFn.NamedInput("lhsI32", shapes.Make(dtypes.Int32)))
		rhsF32 := must1(updateFn.NamedInput("rhsF32", shapes.Make(dtypes.F32)))
		rhsI32 := must1(updateFn.NamedInput("rhsI32", shapes.Make(dtypes.Int32)))
		must(updateFn.Return(
			must1(Add(lhsF32, rhsF32)),
			must1(Add(lhsI32, rhsI32)),
		))
		results := must1(MultiScatter(
			[]*Value{inputsF32, inputsI32}, scatterIndices, []*Value{updatesF32, updatesI32},
			updateWindowAxes, insertedWindowAxes,
			inputBatchingAxes, scatterIndicesBatchingAxes,
			indexedInputAxes, indexVectorAxis,
			false, false,
			updateFn))
		must(fn.Return(results[0], results[1]))
		program := must1(builder.Build())
		fmt.Printf("%s program:\n%s", t.Name(), withLines(program))
		outputs := compileAndExecute(t, client, program)
		requireBuffersEqual(t, []FlatAndDims{
			{[]float32{101, 0, 0, 0, 100, 0, 103, 0, 102, 0, 0, 0}, []int{2, 1, 2, 3}},
			{[]int32{1001, 0, 0, 0, 1000, 0, 1003, 0, 1002, 0, 0, 0}, []int{2, 1, 2, 3}},
		}, outputs)
	})

	t.Run("Convert", func(t *testing.T) {
		builder := New(t.Name())
		fn := builder.Main()
		x0 := must1(fn.Iota(shapes.Make(dtypes.F32, 3), 0))
		x1 := must1(fn.ConstantFromFlatAndDimensions([]bool{true, false, true}, 3))
		must(fn.Return(
			must1(Convert(x0, dtypes.Bool)),
			must1(Convert(x1, dtypes.Int32))))
		program := must1(builder.Build())
		fmt.Printf("%s program:\n%s", t.Name(), withLines(program))
		outputs := compileAndExecute(t, client, program)
		requireBuffersEqual(t, []FlatAndDims{
			{[]bool{false, true, true}, []int{3}},
			{[]int32{1, 0, 1}, []int{3}},
		}, outputs)
	})

	t.Run("Pad", func(t *testing.T) {
		builder := New(t.Name())
		fn := builder.Main()
		x := must1(fn.Iota(shapes.Make(dtypes.F32, 2, 3, 2), 0))
		fill := must1(fn.ConstantFromScalar(float32(3.0)))
		padded := must1(Pad(x, fill, []int{1, 0, 0}, []int{0, 2, 0}, []int{0, 0, 1}))
		must(fn.Return(padded))
		program := must1(builder.Build())
		fmt.Printf("%s program:\n%s", t.Name(), withLines(program))
		outputs := compileAndExecute(t, client, program)
		requireBuffersEqual(t, []FlatAndDims{
			{[]float32{3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 0, 3, 0, 0, 3, 0, 0, 3, 0, 3, 3, 3, 3, 3, 3, 1, 3, 1, 1, 3, 1, 1, 3, 1, 3, 3, 3, 3, 3, 3}, []int{3, 5, 3}},
		}, outputs)
	})

	t.Run("Reverse", func(t *testing.T) {
		builder := New(t.Name())
		fn := builder.Main()
		x := must1(fn.Iota(shapes.Make(dtypes.F32, 3*2), 0))
		x = must1(Reshape(x, shapes.Make(dtypes.F32, 2, 3)))
		must(fn.Return(
			must1(Reverse(x, 0)),
			must1(Reverse(x, 1))))
		program := must1(builder.Build())
		fmt.Printf("%s program:\n%s", t.Name(), withLines(program))
		outputs := compileAndExecute(t, client, program)
		requireBuffersEqual(t, []FlatAndDims{
			{[]float32{3, 4, 5, 0, 1, 2}, []int{2, 3}},
			{[]float32{2, 1, 0, 5, 4, 3}, []int{2, 3}},
		}, outputs)
	})

	t.Run("FFT", func(t *testing.T) {
		builder := New(t.Name())
		fn := builder.Main()
		x := must1(fn.Iota(shapes.Make(dtypes.F32, 3*4*10), 0))
		x = must1(Reshape(x, shapes.Make(dtypes.F32, 3, 4, 10)))
		c := must1(Complex(x, x))
		must(fn.Return(
			must1(FFT(c, types.FFTForward)),
			must1(FFT(c, types.FFTInverse)),
			must1(FFT(x, types.FFTForwardReal)),
			must1(FFT(c, types.FFTInverseReal)),
		))
		program := must1(builder.Build())
		fmt.Printf("%s program:\n%s", t.Name(), withLines(program))
		outputs := compileAndExecute(t, client, program)

		gotDims := must1(outputs[0].Dimensions())
		fmt.Printf("\t- FFTForward output dims: %v\n", gotDims)
		if !reflect.DeepEqual([]int{3, 4, 10}, gotDims) {
			t.Errorf("FFTForward dims mismatch: want %v, got %v", []int{3, 4, 10}, gotDims)
		}

		gotDims = must1(outputs[1].Dimensions())
		fmt.Printf("\t- FFTInverse output dims: %v\n", gotDims)
		if !reflect.DeepEqual([]int{3, 4, 10}, gotDims) {
			t.Errorf("FFTInverse dims mismatch: want %v, got %v", []int{3, 4, 10}, gotDims)
		}

		gotDims = must1(outputs[2].Dimensions())
		gotDType := must1(outputs[2].DType())
		fmt.Printf("\t- FFTForwardReal output dtype %s, dims: %v\n", gotDType, gotDims)
		if !reflect.DeepEqual([]int{3, 4, 10/2 + 1}, gotDims) {
			t.Errorf("FFTForwardReal dims mismatch: want %v, got %v", []int{3, 4, 10/2 + 1}, gotDims)
		}
		if gotDType != dtypes.Complex64 {
			t.Errorf("FFTForwardReal dtype mismatch: want %v, got %v", dtypes.Complex64, gotDType)
		}

		gotDims = must1(outputs[3].Dimensions())
		gotDType = must1(outputs[3].DType())
		fmt.Printf("\t- FFTInverseReal output dtype %s, dims: %v\n", gotDType, gotDims)
		if !reflect.DeepEqual([]int{3, 4, 2 * (10 - 1)}, gotDims) {
			t.Errorf("FFTInverseReal dims mismatch: want %v, got %v", []int{3, 4, 2 * (10 - 1)}, gotDims)
		}
		if gotDType != dtypes.Float32 {
			t.Errorf("FFTInverseReal dtype mismatch: want %v, got %v", dtypes.Float32, gotDType)
		}
	})

	t.Run("ReduceWindow", func(t *testing.T) {
		builder := New(t.Name())
		fn := builder.Main()
		x := must1(fn.Iota(shapes.Make(dtypes.F32, 2*3), 0))
		x = must1(Reshape(x, shapes.Make(dtypes.F32, 2, 3)))
		zero := must1(fn.ConstantFromScalar(float32(0)))
		reductionFn := fn.Closure()
		lhs := must1(reductionFn.NamedInput("lhs", shapes.Make(dtypes.F32)))
		rhs := must1(reductionFn.NamedInput("rhs", shapes.Make(dtypes.F32)))
		must(reductionFn.Return(must1(Add(lhs, rhs))))
		r0 := must1(ReduceWindow(x, zero, reductionFn,
			[]int{2, 2}, []int{1, 1}, nil, nil, nil))
		r1 := must1(ReduceWindow(x, zero, reductionFn,
			[]int{2, 2}, []int{1, 1}, nil, nil, [][2]int{{1, 1}, {1, 1}}))
		must(fn.Return(r0, r1))
		program := must1(builder.Build())
		fmt.Printf("%s program:\n%s", t.Name(), withLines(program))
		outputs := compileAndExecute(t, client, program)
		requireBuffersEqual(t, []FlatAndDims{
			{[]float32{8, 12}, []int{1, 2}},
			{[]float32{
				0, 1, 3, 2,
				3, 8, 12, 7,
				3, 7, 9, 5}, []int{3, 4}},
		}, outputs)
	})

	t.Run("SelectAndScatter", func(t *testing.T) {
		builder := New(t.Name())
		fn := builder.Main()
		x := must1(fn.Iota(shapes.Make(dtypes.F32, 2*3), 0))
		x = must1(Reshape(x, shapes.Make(dtypes.F32, 2, 3)))
		one := must1(fn.ConstantFromScalar(float32(1)))
		source0 := must1(BroadcastInDim(one, shapes.Make(dtypes.F32, 1, 2), nil))
		source1 := must1(BroadcastInDim(one, shapes.Make(dtypes.F32, 3, 4), nil))

		selectFn := fn.Closure() // return lhs >= rhs  --> it will select the max of the window.
		{
			lhs := must1(selectFn.NamedInput("lhs", shapes.Make(dtypes.F32)))
			rhs := must1(selectFn.NamedInput("rhs", shapes.Make(dtypes.F32)))
			must(selectFn.Return(must1(Compare(lhs, rhs, types.CompareGE, types.CompareFloat))))
		}
		scatterFn := fn.Closure() // return lhs+rhs  --> it will sum all contributions to the location.
		{
			lhs := must1(scatterFn.NamedInput("lhs", shapes.Make(dtypes.F32)))
			rhs := must1(scatterFn.NamedInput("rhs", shapes.Make(dtypes.F32)))
			must(scatterFn.Return(must1(Add(lhs, rhs))))
		}

		seven := must1(fn.ConstantFromScalar(float32(7)))
		mil := must1(fn.ConstantFromScalar(float32(1000)))

		r0 := must1(SelectAndScatter(x, source0, seven, selectFn, scatterFn,
			[]int{2, 2}, []int{1, 1}, nil))
		r1 := must1(SelectAndScatter(x, source1, mil, selectFn, scatterFn,
			[]int{2, 2}, []int{1, 1}, [][2]int{{1, 1}, {1, 1}}))
		must(fn.Return(r0, r1))
		program := must1(builder.Build())
		fmt.Printf("%s program:\n%s", t.Name(), withLines(program))
		outputs := compileAndExecute(t, client, program)
		requireBuffersEqual(t, []FlatAndDims{
			{[]float32{7, 7, 7, 7, 8, 8}, []int{2, 3}},
			{[]float32{1001, 1001, 1002, 1002, 1002, 1004}, []int{2, 3}}},
			outputs)
	})

	t.Run("DynamicSlice", func(t *testing.T) {
		builder := New(t.Name())
		fn := builder.Main()
		x := must1(fn.Iota(shapes.Make(dtypes.F32, 3*3), 0))
		x = must1(Reshape(x, shapes.Make(dtypes.F32, 3, 3)))
		one := must1(fn.ConstantFromScalar(int32(1)))
		minusOne := must1(fn.ConstantFromScalar(int32(-1)))
		must(fn.Return(must1(
			DynamicSlice(x, []*Value{minusOne, one}, []int{2, 2}))))
		program := must1(builder.Build())
		fmt.Printf("%s program:\n%s", t.Name(), withLines(program))
		outputs := compileAndExecute(t, client, program)
		requireBuffersEqual(t, []FlatAndDims{
			// Notice that because of the clamp imposed by DynamicSlice, the startIndices are moved to {0, 1}:
			{[]float32{1, 2, 4, 5}, []int{2, 2}},
		}, outputs)
	})

	t.Run("DynamicUpdateSlice", func(t *testing.T) {
		builder := New(t.Name())
		fn := builder.Main()
		x := must1(fn.Iota(shapes.Make(dtypes.F32, 3*3), 0))
		x = must1(Reshape(x, shapes.Make(dtypes.F32, 3, 3)))
		mil := must1(fn.ConstantFromScalar(float32(1000)))
		update := must1(fn.Iota(shapes.Make(dtypes.F32, 2*2), 0))
		update = must1(Reshape(update, shapes.Make(dtypes.F32, 2, 2)))
		update = must1(Add(
			must1(BroadcastInDim(mil, shapes.Make(dtypes.F32, 2, 2), nil)),
			update))
		one := must1(fn.ConstantFromScalar(int32(1)))
		minusOne := must1(fn.ConstantFromScalar(int32(-1)))
		must(fn.Return(must1(
			DynamicUpdateSlice(x, update, []*Value{minusOne, one}))))
		program := must1(builder.Build())
		fmt.Printf("%s program:\n%s", t.Name(), withLines(program))
		outputs := compileAndExecute(t, client, program)
		requireBuffersEqual(t, []FlatAndDims{
			// Notice that because of the clamp imposed by DynamicSlice, the startIndices are moved to {0, 1}:
			{[]float32{
				0, 1000, 1001,
				3, 1002, 1003,
				6, 7, 8}, []int{3, 3}},
		}, outputs)
	})

	t.Run("BatchNormInference", func(t *testing.T) {
		builder := New(t.Name())
		fn := builder.Main()
		x := must1(fn.Iota(shapes.Make(dtypes.F32, 7, 3), 0))
		scale := must1(fn.ConstantFromFlatAndDimensions([]float32{1, 2, 3}, 3))
		offset := must1(fn.ConstantFromFlatAndDimensions([]float32{10, 100, 1000}, 3))
		mean := must1(fn.ConstantFromFlatAndDimensions([]float32{0.5, 0.5, 1}, 3))
		variance := must1(fn.ConstantFromFlatAndDimensions([]float32{1, 1, 10}, 3))
		must(fn.Return(must1(
			BatchNormInference(x, scale, offset, mean, variance, 1e-7, -1))))
		program := must1(builder.Build())
		fmt.Printf("%s program:\n%s", t.Name(), withLines(program))
		outputs := compileAndExecute(t, client, program)
		requireBuffersEqual(t, []FlatAndDims{
			// Notice that because of the clamp imposed by DynamicSlice, the startIndices are moved to {0, 1}:
			{[]float32{
				9.5, 99, 999.05133,
				10.5, 101, 1000,
				11.5, 103, 1000.94867,
				12.5, 105, 1001.8974,
				13.5, 107, 1002.84607,
				14.5, 109, 1003.79474,
				15.5, 111, 1004.7434,
			}, []int{7, 3}},
		}, outputs)
	})

	t.Run("BatchNormTraining", func(t *testing.T) {
		builder := New(t.Name())
		fn := builder.Main()
		x := must1(fn.Iota(shapes.Make(dtypes.F32, 7, 3), 0))
		scale := must1(fn.ConstantFromFlatAndDimensions([]float32{1, 2, 3}, 3))
		offset := must1(fn.ConstantFromFlatAndDimensions([]float32{10, 100, 1000}, 3))
		xNorm, batchMean, batchVariance, err := BatchNormTraining(x, scale, offset, 1e-7, -1)
		if err != nil {
			t.Fatalf("BatchNormTraining error: %v", err)
		}
		must(fn.Return(xNorm, batchMean, batchVariance))
		program := must1(builder.Build())
		fmt.Printf("%s program:\n%s", t.Name(), withLines(program))
		outputs := compileAndExecute(t, client, program)
		requireBuffersEqual(t, []FlatAndDims{
			// Notice that because of the clamp imposed by DynamicSlice, the startIndices are moved to {0, 1}:
			{[]float32{
				8.5, 97, 995.5,
				9, 98, 997,
				9.5, 99, 998.5,
				10, 100, 1000,
				10.5, 101, 1001.5,
				11, 102, 1003,
				11.5, 103, 1004.5,
			}, []int{7, 3}},
			{[]float32{3, 3, 3}, []int{3}}, // Mean = (0+1+2+3+4+5+6) / 7 = 3
			{[]float32{4, 4, 4}, []int{3}}, // ReduceVariance = (9+4+1+0+1+4+9) / 7 = 4
		}, outputs)
	})

	t.Run("BatchNormGradient", func(t *testing.T) {
		builder := New(t.Name())
		fn := builder.Main()
		x := must1(fn.Iota(shapes.Make(dtypes.F32, 7, 3), 0))
		scale := must1(fn.ConstantFromFlatAndDimensions([]float32{1, 2, 3}, 3))
		mean := must1(fn.ConstantFromFlatAndDimensions([]float32{0.5, 0.5, 1}, 3))
		variance := must1(fn.ConstantFromFlatAndDimensions([]float32{1, 1, 10}, 3))
		gradOutput := must1(fn.ConstantFromScalar(float32(1)))
		gradOutput = must1(BroadcastInDim(gradOutput, shapes.Make(dtypes.F32, 7, 3), nil))
		gradX, gradScale, gradOffset, err := BatchNormGradient(x, scale, mean, variance, gradOutput, 1e-7, -1)
		if err != nil {
			t.Fatalf("BatchNormGradient error: %v", err)
		}
		must(fn.Return(gradX, gradScale, gradOffset))
		program := must1(builder.Build())
		fmt.Printf("%s program:\n%s", t.Name(), withLines(program))
		outputs := compileAndExecute(t, client, program)
		requireBuffersEqual(t, []FlatAndDims{
			// Notice that because of the clamp imposed by DynamicSlice, the startIndices are moved to {0, 1}:
			{[]float32{
				1.2500, 2.5000, 0.1897,
				-1.2500, -2.5000, 0,
				-3.7500, -7.5000, -0.1897,
				-6.2500, -12.5000, -0.3795,
				-8.7500, -17.5000, -0.5692,
				-11.2500, -22.5000, -0.7589,
				-13.7500, -27.5000, -0.94868326,
			}, []int{7, 3}},
			{[]float32{17.5, 17.5, 4.427189}, []int{3}}, // Gradient wrt. scale, affected by the mean.
			{[]float32{7, 7, 7}, []int{3}},              // The offset impacts each feature equally.
		}, outputs)
	})
}

func TestBinaryOps(t *testing.T) {
	iterateClientsAndTest(t, testBinaryOps)
}

func testBinaryOps(t *testing.T, client *pjrt.Client) {
	testBinaryOp := func(t *testing.T, opName string,
		op func(lhs, rhs *Value) (*Value, error),
		dtype dtypes.DType, lhs, rhs any, expected any) {
		builder := New(t.Name())
		shape := shapes.Make(dtype)
		fn := builder.Main()
		lhsV, rhsV := must1(fn.NamedInput("lhs", shape)), must1(fn.NamedInput("rhs", shape))
		result := must1(op(lhsV, rhsV))
		must(fn.Return(result))
		program := must1(builder.Build())
		fmt.Printf("%s program:\n%s", t.Name(), withLines(program))
		a := must1(client.BufferFromHost().FromFlatDataWithDimensions(lhs, []int{}).Done())
		b := must1(client.BufferFromHost().FromFlatDataWithDimensions(rhs, []int{}).Done())
		output := compileAndExecute(t, client, program, a, b)
		requireBuffersEqual(t, []FlatAndDims{{expected, nil}}, output)
	}

	t.Run("Add", func(t *testing.T) {
		testBinaryOp(t, "Add", Add, dtypes.Float32, []float32{3.0}, []float32{7.0}, []float32{10.0})
	})
	t.Run("Atan2", func(t *testing.T) {
		testBinaryOp(t, "Atan2", Atan2, dtypes.Float32, []float32{3.0}, []float32{7.0},
			[]float32{float32(math.Atan2(3.0, 7.0))})
	})

	t.Run("Subtract", func(t *testing.T) {
		testBinaryOp(t, "Subtract", Subtract, dtypes.Float32, []float32{7.0}, []float32{3.0}, []float32{4.0})
	})

	t.Run("Multiply", func(t *testing.T) {
		testBinaryOp(t, "Multiply", Multiply, dtypes.Float32, []float32{3.0}, []float32{4.0}, []float32{12.0})
	})

	t.Run("Divide", func(t *testing.T) {
		testBinaryOp(t, "Divide", Divide, dtypes.Float32, []float32{12.0}, []float32{3.0}, []float32{4.0})
	})

	t.Run("Power", func(t *testing.T) {
		testBinaryOp(t, "Power", Power, dtypes.Float32, []float32{2.0}, []float32{3.0}, []float32{8.0})
	})

	t.Run("And_Uint32", func(t *testing.T) {
		testBinaryOp(t, "And", And, dtypes.Uint32, []uint32{0b1100}, []uint32{0b1010}, []uint32{0b1000})
	})

	t.Run("Or_Uint32", func(t *testing.T) {
		testBinaryOp(t, "Or", Or, dtypes.Uint32, []uint32{0b1100}, []uint32{0b1010}, []uint32{0b1110})
	})

	t.Run("Xor_Uint32", func(t *testing.T) {
		testBinaryOp(t, "Xor", Xor, dtypes.Uint32, []uint32{0b1100}, []uint32{0b1010}, []uint32{0b0110})
	})

	t.Run("And_Bool", func(t *testing.T) {
		testBinaryOp(t, "And", And, dtypes.Bool, []bool{true}, []bool{false}, []bool{false})
	})

	t.Run("Or_Bool", func(t *testing.T) {
		testBinaryOp(t, "Or", Or, dtypes.Bool, []bool{true}, []bool{false}, []bool{true})
	})

	t.Run("Xor_Bool", func(t *testing.T) {
		testBinaryOp(t, "Xor", Xor, dtypes.Bool, []bool{true}, []bool{false}, []bool{true})
	})

	t.Run("ShiftLeft", func(t *testing.T) {
		testBinaryOp(t, "ShiftLeft", ShiftLeft, dtypes.Uint32, []uint32{0b1}, []uint32{2}, []uint32{0b100})
	})

	t.Run("ShiftRightArithmetic", func(t *testing.T) {
		testBinaryOp(t, "ShiftRightArithmetic", ShiftRightArithmetic, dtypes.Int32, []int32{-8}, []int32{1}, []int32{-4})
	})

	t.Run("ShiftRightLogical", func(t *testing.T) {
		testBinaryOp(t, "ShiftRightLogical", ShiftRightLogical, dtypes.Uint32, []uint32{0b1100}, []uint32{2}, []uint32{0b11})
	})

	t.Run("Reminder", func(t *testing.T) {
		testBinaryOp(t, "Remainder", Remainder, dtypes.Float32, []float32{7.0}, []float32{4.0}, []float32{3.0})
	})

	t.Run("Maximum", func(t *testing.T) {
		testBinaryOp(t, "Maximum", Maximum, dtypes.Float32, []float32{3.0}, []float32{7.0}, []float32{7.0})
	})

	t.Run("Minimum", func(t *testing.T) {
		testBinaryOp(t, "Minimum", Minimum, dtypes.Float32, []float32{3.0}, []float32{7.0}, []float32{3.0})
	})

}

func TestCompare(t *testing.T) {
	iterateClientsAndTest(t, testCompare)
}

func testCompare(t *testing.T, client *pjrt.Client) {
	runTest := func(t *testing.T, opName string,
		direction types.ComparisonDirection, compareType types.ComparisonType,
		dtype dtypes.DType, lhs, rhs any, expected any) {
		builder := New(t.Name())
		var dims []int
		if reflect.ValueOf(lhs).Len() > 1 {
			dims = []int{reflect.ValueOf(lhs).Len()}
		}
		shape := shapes.Make(dtype, dims...)
		fn := builder.Main()
		lhsV, rhsV := must1(fn.NamedInput("lhs", shape)), must1(fn.NamedInput("rhs", shape))
		result := must1(Compare(lhsV, rhsV, direction, compareType))
		must(fn.Return(result))
		program := must1(builder.Build())
		fmt.Printf("%s program:\n%s", t.Name(), withLines(program))
		a := must1(client.BufferFromHost().FromFlatDataWithDimensions(lhs, dims).Done())
		b := must1(client.BufferFromHost().FromFlatDataWithDimensions(rhs, dims).Done())
		output := compileAndExecute(t, client, program, a, b)
		requireBuffersEqual(t, []FlatAndDims{{expected, dims}}, output)
	}

	t.Run("Float_EQ", func(t *testing.T) {
		runTest(t, "Compare", types.CompareEQ, types.CompareFloat,
			dtypes.Float32, []float32{3.0}, []float32{3.0}, []bool{true})
	})

	t.Run("Signed_GT", func(t *testing.T) {
		runTest(t, "Compare", types.CompareGT, types.CompareSigned,
			dtypes.Int32, []int32{7}, []int32{3}, []bool{true})
	})

	t.Run("Unsigned_LT", func(t *testing.T) {
		runTest(t, "Compare", types.CompareLT, types.CompareUnsigned,
			dtypes.Uint32, []uint32{3}, []uint32{7}, []bool{true})
	})

	t.Run("TotalOrder_GE", func(t *testing.T) {
		runTest(t, "Compare", types.CompareGE, types.CompareTotalOrder,
			dtypes.Float32, []float32{3.0}, []float32{3.0}, []bool{true})
	})

	t.Run("Float_NE", func(t *testing.T) {
		runTest(t, "Compare", types.CompareNE, types.CompareFloat,
			dtypes.Float32, []float32{3.0}, []float32{7.0}, []bool{true})
	})

	t.Run("Signed_LE", func(t *testing.T) {
		runTest(t, "Compare", types.CompareLE, types.CompareSigned,
			dtypes.Int32, []int32{3}, []int32{7}, []bool{true})
	})

	t.Run("Bool_EQ", func(t *testing.T) {
		runTest(t, "Compare", types.CompareEQ, types.CompareUnsigned,
			dtypes.Bool,
			[]bool{true, true, false, false},
			[]bool{true, false, true, false},
			[]bool{true, false, false, true})
	})

	t.Run("Bool_NE", func(t *testing.T) {
		runTest(t, "Compare", types.CompareNE, types.CompareUnsigned,
			dtypes.Bool,
			[]bool{true, true, false, false},
			[]bool{true, false, true, false},
			[]bool{false, true, true, false})
	})

	t.Run("Bool_GT", func(t *testing.T) {
		runTest(t, "Compare", types.CompareGT, types.CompareUnsigned,
			dtypes.Bool,
			[]bool{true, true, false, false},
			[]bool{true, false, true, false},
			[]bool{false, true, false, false})
	})

}

const pi32 = float32(math.Pi)

func TestUnaryOps(t *testing.T) {
	iterateClientsAndTest(t, testUnaryOps)
}

func testUnaryOps(t *testing.T, client *pjrt.Client) {
	testUnaryOp := func(t *testing.T, opName string,
		op func(x *Value) (*Value, error),
		dtype dtypes.DType, input any, expected any) {
		builder := New(t.Name())
		shape := shapes.Make(dtype)
		fn := builder.Main()
		arg, err := fn.Input(shape)
		if err != nil {
			t.Fatalf("fn.Input error: %v", err)
		}
		result := must1(op(arg))
		must(fn.Return(result))
		program := must1(builder.Build())
		fmt.Printf("%s program:\n%s", t.Name(), withLines(program))
		a := must1(client.BufferFromHost().FromFlatDataWithDimensions(input, []int{}).Done())
		output := compileAndExecute(t, client, program, a)
		requireBuffersEqual(t, []FlatAndDims{{expected, nil}}, output)
	}

	t.Run("Not_Bool", func(t *testing.T) {
		testUnaryOp(t, "Not", Not, dtypes.Bool, []bool{true}, []bool{false})
	})
	t.Run("Not_Uint8", func(t *testing.T) {
		testUnaryOp(t, "Not", Not, dtypes.Uint8, []uint8{128}, []uint8{127})
	})

	t.Run("Popcnt_Uint32", func(t *testing.T) {
		testUnaryOp(t, "Popcnt", Popcnt, dtypes.Uint32, []uint32{0b1011}, []uint32{3})
	})

	t.Run("CountLeadingZeros_Uint32", func(t *testing.T) {
		testUnaryOp(t, "CountLeadingZeros", CountLeadingZeros, dtypes.Uint32, []uint32{0b1}, []uint32{31})
	})

	t.Run("Erf_Float32", func(t *testing.T) {
		testUnaryOp(t, "Erf", Erf, dtypes.Float64, []float64{1.0}, []float64{
			math.Erf(1)})
	})

	t.Run("Exponential_Float32", func(t *testing.T) {
		testUnaryOp(t, "Exponential", Exponential, dtypes.Float32, []float32{1.0},
			[]float32{float32(math.Exp(1))})
	})

	t.Run("ExponentialMinusOne_Float32", func(t *testing.T) {
		testUnaryOp(t, "ExponentialMinusOne", ExponentialMinusOne, dtypes.Float32, []float32{1.0},
			[]float32{float32(math.E - 1)})
	})

	t.Run("Log_Float32", func(t *testing.T) {
		testUnaryOp(t, "Log", Log, dtypes.Float32, []float32{2.7183}, []float32{1.0})
	})

	t.Run("LogPlusOne_Float32", func(t *testing.T) {
		testUnaryOp(t, "LogPlusOne", LogPlusOne, dtypes.Float32, []float32{1.7183}, []float32{1.0})
	})

	t.Run("Logistic_Float32", func(t *testing.T) {
		testUnaryOp(t, "Logistic", Logistic, dtypes.Float32, []float32{0.0}, []float32{0.5})
	})

	t.Run("Ceil_Float32", func(t *testing.T) {
		testUnaryOp(t, "Ceil", Ceil, dtypes.Float32, []float32{1.7}, []float32{2.0})
	})

	t.Run("Floor_Float32", func(t *testing.T) {
		testUnaryOp(t, "Floor", Floor, dtypes.Float32, []float32{1.7}, []float32{1.0})
	})

	t.Run("RoundNearestEven_Float32", func(t *testing.T) {
		testUnaryOp(t, "RoundNearestEven", RoundNearestEven, dtypes.Float32, []float32{2.5}, []float32{2.0})
	})
	t.Run("RoundNearestAfz_Float32", func(t *testing.T) {
		testUnaryOp(t, "RoundNearestAfz", RoundNearestAfz, dtypes.Float32, []float32{2.5}, []float32{3.0})
	})

	t.Run("Rsqrt_Float32", func(t *testing.T) {
		testUnaryOp(t, "Rsqrt", Rsqrt, dtypes.Float32, []float32{4.0}, []float32{0.5})
	})

	t.Run("Sqrt_Float32", func(t *testing.T) {
		testUnaryOp(t, "Sqrt", Sqrt, dtypes.Float32, []float32{4.0}, []float32{2.0})
	})

	t.Run("Cbrt_Float32", func(t *testing.T) {
		testUnaryOp(t, "Cbrt", Cbrt, dtypes.Float32, []float32{8.0}, []float32{2.0})
	})

	t.Run("Cosine_Float32", func(t *testing.T) {
		testUnaryOp(t, "Cosine", Cosine, dtypes.Float32, []float32{0.0}, []float32{1.0})
	})

	t.Run("Sine_Float32", func(t *testing.T) {
		testUnaryOp(t, "Sine", Sine, dtypes.Float32, []float32{0.0}, []float32{0.0})
	})

	t.Run("Tan_Float32", func(t *testing.T) {
		testUnaryOp(t, "Tan", Tan, dtypes.Float32, []float32{pi32 / 4}, []float32{1})
	})
	t.Run("Tanh_Float32", func(t *testing.T) {
		testUnaryOp(t, "Tanh", Tanh, dtypes.Float32, []float32{0.5},
			[]float32{float32(math.Tanh(0.5))})
	})

	t.Run("Abs_Float32", func(t *testing.T) {
		testUnaryOp(t, "Abs", Abs, dtypes.Float32, []float32{-3.0}, []float32{3.0})
	})

	t.Run("Abs_Complex64", func(t *testing.T) {
		testUnaryOp(t, "Abs", Abs, dtypes.Complex64, []complex64{complex64(-3.0 + 4i)}, []float32{5.0})
	})

	t.Run("Negate_Float32", func(t *testing.T) {
		testUnaryOp(t, "Negate", Negate, dtypes.Float32, []float32{3.0}, []float32{-3.0})
	})

	t.Run("Sign_Float32", func(t *testing.T) {
		testUnaryOp(t, "Sign", Sign, dtypes.Float32, []float32{-3.0}, []float32{-1.0})
	})

	t.Run("Real_Complex64", func(t *testing.T) {
		testUnaryOp(t, "Real", Real, dtypes.Complex64, []complex64{complex(3.0, 4.0)}, []float32{3.0})
	})

	t.Run("Real_Complex128", func(t *testing.T) {
		testUnaryOp(t, "Real", Real, dtypes.Complex128, []complex128{complex(3.0, 4.0)}, []float64{3.0})
	})

	t.Run("Imag_Complex64", func(t *testing.T) {
		testUnaryOp(t, "Imag", Imag, dtypes.Complex64, []complex64{complex(3.0, 4.0)}, []float32{4.0})
	})

	t.Run("Imag_Complex128", func(t *testing.T) {
		testUnaryOp(t, "Imag", Imag, dtypes.Complex128, []complex128{complex(3.0, 4.0)}, []float64{4.0})
	})
}

func TestConstants(t *testing.T) {
	iterateClientsAndTest(t, testConstants)
}

func testConstants(t *testing.T, client *pjrt.Client) {
	testScalar := func(t *testing.T, scalar any) {
		builder := New(t.Name())
		fn := builder.Main()
		c, err := fn.ConstantFromScalar(scalar)
		if err != nil {
			t.Fatalf("ConstantFromScalar error: %v", err)
		}
		must(fn.Return(c))
		program := must1(builder.Build())
		fmt.Printf("%s program:\n%s", t.Name(), withLines(program))
		output := compileAndExecute(t, client, program)[0]
		gotFlat, gotDim, err := output.ToFlatDataAndDimensions()
		if err != nil {
			t.Fatalf("ToFlatDataAndDimensions error: %v", err)
		}
		if len(gotDim) != 0 {
			t.Errorf("gotDim len %d, want 0", len(gotDim))
		}
		gotScalar := reflect.ValueOf(gotFlat).Index(0).Interface()
		if !reflect.DeepEqual(scalar, gotScalar) {
			t.Errorf("gotScalar %v, want %v", gotScalar, scalar)
		}
	}

	t.Run("float32", func(t *testing.T) { testScalar(t, float32(3.0)) })
	t.Run("float64", func(t *testing.T) { testScalar(t, 1.234e-9) })
	t.Run("int64", func(t *testing.T) { testScalar(t, int64(-3)) })
	t.Run("uint8", func(t *testing.T) { testScalar(t, uint8(3)) })
	t.Run("bool-true", func(t *testing.T) { testScalar(t, true) })
	t.Run("bool-false", func(t *testing.T) { testScalar(t, false) })
	t.Run("complex64", func(t *testing.T) { testScalar(t, complex64(7-3i)) })
	t.Run("complex128", func(t *testing.T) { testScalar(t, complex64(-7+3i)) })

	testTensor := func(t *testing.T, flat any, dimensions ...int) {
		builder := New(t.Name())
		fn := builder.Main()
		c, err := fn.ConstantFromFlatAndDimensions(flat, dimensions...)
		if err != nil {
			t.Fatalf("ConstantFromFlatAndDimensions error: %v", err)
		}
		must(fn.Return(c))
		program := must1(builder.Build())
		fmt.Printf("%s program:\n%s", t.Name(), withLines(program))
		output := compileAndExecute(t, client, program)[0]
		gotFlat, gotDims, err := output.ToFlatDataAndDimensions()
		if err != nil {
			t.Fatalf("ToFlatDataAndDimensions error: %v", err)
		}
		if !reflect.DeepEqual(dimensions, gotDims) {
			// Handle nil vs empty slice for dims
			if len(dimensions) != 0 || len(gotDims) != 0 {
				t.Errorf("dims mismatch: want %v, got %v", dimensions, gotDims)
			}
		}
		if !reflect.DeepEqual(flat, gotFlat) {
			t.Errorf("flat data mismatch: want %v, got %v", flat, gotFlat)
		}
	}

	t.Run("0D-int8", func(t *testing.T) { testTensor(t, []int8{-3}) })
	t.Run("1D-float32", func(t *testing.T) { testTensor(t, []float32{1, 2, 3, 5, 7}, 5) })
	t.Run("2D-complex64", func(t *testing.T) { testTensor(t, []complex64{1, 2, 3, 5i, 7i, 11i}, 2, 3) })
	t.Run("3D-bool", func(t *testing.T) { testTensor(t, []bool{false, true, false, true}, 2, 1, 2) })
}

func TestCall(t *testing.T) {
	iterateClientsAndTest(t, func(t *testing.T, client *pjrt.Client) {
		t.Run("pjrt-only", func(t *testing.T) {
			program := []byte(`module @TestCall_simple {
		func.func @double(%arg0: tensor<f32>) -> tensor<f32> {
			%0 = "stablehlo.add"(%arg0, %arg0) : (tensor<f32>, tensor<f32>) -> tensor<f32>
			"stablehlo.return"(%0) : (tensor<f32>) -> ()
		}

		func.func @main(%x: tensor<f32>) -> tensor<f32> {
			%0 = "func.call"(%x) {callee = @double} : (tensor<f32>) -> tensor<f32>
			"stablehlo.return"(%0) : (tensor<f32>) -> ()
		}
}`)
			// Compile
			loadedExec := must1(client.Compile().WithStableHLO(program).Done())
			defer loadedExec.Destroy()

			// Execute
			inputVal := float32(10.0)
			inputBuffer := must1(pjrt.ScalarToBufferOnDeviceNum(client, 0, inputVal))
			defer inputBuffer.Destroy()

			outputBuffers := must1(loadedExec.Execute(inputBuffer).Done())
			if len(outputBuffers) != 1 {
				t.Fatalf("expected 1 output buffer, got %d", len(outputBuffers))
			}
			defer outputBuffers[0].Destroy()
			outputVal := must1(pjrt.BufferToScalar[float32](outputBuffers[0]))

			expected := inputVal * 2
			if outputVal != expected {
				t.Errorf("expected %f, got %f", expected, outputVal)
			}
		})

		t.Run("simple", func(t *testing.T) {
			b := stablehlo.New(t.Name())

			// Define callee: double(x) = x + x
			callee := b.NewFunction("double")
			arg := must1(callee.Input(shapes.Make(dtypes.F32)))
			doubleArg := must1(stablehlo.Add(arg, arg))
			must(callee.Return(doubleArg))

			// Define main: main(x) = double(x)
			fn := b.Main()
			input := must1(fn.NamedInput("x", shapes.Make(dtypes.F32)))
			results := must1(stablehlo.Call(callee, input))
			must(fn.Return(results...))

			// Build
			program := must1(b.Build())
			fmt.Printf("%s program:\n%s", t.Name(), program)

			// Compile
			loadedExec := must1(client.Compile().WithStableHLO(program).Done())
			defer loadedExec.Destroy()

			// Execute
			inputVal := float32(10.0)
			inputBuffer := must1(pjrt.ScalarToBufferOnDeviceNum(client, 0, inputVal))
			defer inputBuffer.Destroy()

			outputBuffers := must1(loadedExec.Execute(inputBuffer).Done())
			if len(outputBuffers) != 1 {
				t.Fatalf("expected 1 output buffer, got %d", len(outputBuffers))
			}
			defer outputBuffers[0].Destroy()
			outputVal := must1(pjrt.BufferToScalar[float32](outputBuffers[0]))

			expected := inputVal * 2
			if outputVal != expected {
				t.Errorf("expected %f, got %f", expected, outputVal)
			}
		})
	})
}
