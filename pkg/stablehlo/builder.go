package stablehlo

import (
	"bytes"
	"fmt"
	"io"
	"slices"

	"github.com/gomlx/go-xla/internal/utils"
	"github.com/gomlx/go-xla/pkg/types"
	"github.com/gomlx/go-xla/pkg/types/shardy"
	"github.com/pkg/errors"
)

// Builder is used to construct a StableHLO program (or "Module")
// See details in New.
type Builder struct {
	name   string
	parent *Builder

	// functions holds all the functions created in the builder's scope.
	functions []*Function

	// inlineUniqueID is a counter used to generate unique names for inlined functions values.
	inlineUniqueID int

	// meshes used for Shardy.
	meshes []*shardy.DeviceMesh

	// numReplicas is the number of replicas for data parallelism.
	numReplicas int

	// numPartitions is the number of partitions for model parallelism.
	numPartitions int

	// nextChannelID is the next ID to be assigned in channel handles.
	// It is just a Unique ID.
	nextChannelID int
}

// New creates a new Builder object holding a computation graph in construction.
//
// From a builder you can create functions.
// For each function you create operations (ops) one by one, until you defined the desired computation.
//
// You have to define the "main" function for your StableHLO program: you can use Builder.Main to do so, or
// Builder.NewFunction("main",...), it's the same.
//
// Once you are all set, call Builder.Build and it will return the StableHLO program (or "Module") as a []byte that can
// be used with PJRT.
//
// See github.com/gomlx/go-xla for a Go API to PJRT.
func New(name string) *Builder {
	return &Builder{
		name: name,
	}
}

// elementWriter represents elements of ToStableHLO that know how to write themselves.
type elementWriter interface {
	Write(w io.Writer, indentation string) error
}

// NewFunction creates a new function and adds it to the program.
// The function outputs will be determined by the last statement in the function body.
//
// The function name must be unique in the program.
//
// The inputs are the values that the function will receive as arguments.
// The values are not added to the program, they are just used as inputs.
//
// You can also add new inputs later by calling Function.Input.
//
// The function body is defined by calling ops on the function object.
//
// See Function.
func (b *Builder) NewFunction(name string, inputs ...*Value) *Function {
	fn := &Function{
		Builder: b,
		Name:    name,
		Inputs:  inputs,
		values:  slices.Clone(inputs),
	}
	b.functions = append(b.functions, fn)
	return fn
}

const MainFunctionName = "main"

// Main creates the main function of the program.
// It is an alias to Builder.NewFunction("main", inputs...).
//
// The main function is the entry point of the program, and it's the only function that can be called from outside the program.
//
// Every program must have a main function.
//
// Like with NewFunction, you can add new inputs later by calling Function.Input.
func (b *Builder) Main(inputs ...*Value) *Function {
	return b.NewFunction(MainFunctionName, inputs...)
}

const IndentationStep = "  "

// getModuleAttributes returns the attributes for the StableHLO module (StableHLO code) generated.
func (b *Builder) getModuleAttributes() []string {
	var attributes []string
	if b.numReplicas > 0 {
		attributes = append(attributes, fmt.Sprintf("stablehlo.num_replicas = %d", b.numReplicas))
	}
	if b.numPartitions > 0 {
		attributes = append(attributes, fmt.Sprintf(" stablehlo.num_partitions = %d", b.numPartitions))
	}
	return attributes
}

// Write the StableHLO program (a readable string) to the given writer.
//
// It will write incomplete programs (without a main function or empty statements) without an error
// to help debugging.
//
// See Builder.Build to check and output the program.
func (b *Builder) Write(writer io.Writer) error {
	var err error
	w := func(format string, args ...any) {
		if err != nil {
			// No op if an error was encountered earlier
			return
		}
		_, err = fmt.Fprintf(writer, format, args...)
	}
	we := func(e elementWriter, indentation string) {
		if err != nil {
			// No op if an error was encountered earlier
			return
		}
		err = e.Write(writer, indentation)
	}

	// Write module header
	w("module @%s", NormalizeIdentifier(b.name))
	attrs := b.getModuleAttributes()
	if len(attrs) > 0 {
		w(" attributes {")
		for i, attr := range attrs {
			if i > 0 {
				w(", ")
			}
			w("%s", attr)
		}
		w("}")
	}
	w(" {\n")

	// Write Shardy meshes if needed:
	if len(b.meshes) > 0 {
		namesUsed := utils.MakeSet[string](len(b.meshes))
		for _, mesh := range b.meshes {
			if namesUsed.Has(mesh.Name()) {
				return errors.Errorf("duplicate mesh name %q", mesh.Name())
			}
			namesUsed.Insert(mesh.Name())
			w("%s%s\n", IndentationStep, mesh.ToStableHLO())
		}
	}

	// Write non-inline functions:
	var count int
	for _, fn := range b.functions {
		if fn.Parent != nil {
			continue
		}
		if count > 0 {
			w("\n\n")
		}
		we(fn, IndentationStep) // Indent functions inside module
		count++
	}
	w("\n}\n") // Close module block
	return err
}

// Build checks the validity and builds the StableHLO program.
//
// If you want the output of an incomplete program (without the checking), use Builder.Write instead.
func (b *Builder) Build() ([]byte, error) {
	hasMain := false
	for _, fn := range b.functions {
		if fn.Name == "main" {
			hasMain = true
		}
		if len(fn.Statements) == 0 {
			return nil, fmt.Errorf("function %q has no statements", fn.Name)
		}
	}
	if !hasMain {
		return nil, errors.New("program must have a main function")
	}

	var buf bytes.Buffer
	err := b.Write(&buf)
	if err != nil {
		return nil, err
	}
	return buf.Bytes(), nil
}

// getChannelHandle generates the channel_handle attribute string.
// It uses the config if provided (for MPMD), or the builder's internal
// counter if not (for SPMD).
func (b *Builder) getChannelHandle(config *types.CollectiveConfig) literalStr {
	var id int
	var typ int64

	if config != nil {
		typ = int64(config.ChannelType) // Use specified type
		if config.ChannelID != nil {
			// Manual ID provided (MPMD case)
			id = *config.ChannelID
		} else {
			// Automatic ID (SPMD case)
			id = b.nextChannelID
			b.nextChannelID++
		}
	} else {
		// Defaults for the simple SPMD case.
		typ = int64(types.CrossReplica)
		id = b.nextChannelID
		b.nextChannelID++
	}

	return literalStrF("#stablehlo.channel_handle<handle = %d, type = %d>", id, typ)
}

// WithNumReplicas sets the number of replicas (for data parallelism).
// This is added as an attribute to the StableHLO module.
//
// Consider using WithShardy for distributed computation instead: other forms of distributed
// (collective) computation across devices are not tested and may not work.
func (b *Builder) WithNumReplicas(n int) *Builder {
	b.numReplicas = n
	return b
}

// WithNumPartitions sets the number of partitions (for model parallelism).
// This is added as an attribute to the StableHLO module.
//
// Consider using WithShardy for distributed computation instead: other forms of distributed
// (collective) computation across devices are not tested and may not work.
func (b *Builder) WithNumPartitions(n int) *Builder {
	b.numPartitions = n
	return b
}

// WithShardy enables distributed computation across the devices selected by the given meshes.
//
// This is the recommended way to do distributed (across devices) computation, and given the inputs
// with sharded information, Shardy will automatically distribute the computation, without you needing
// to specify any of the collective operations.
//
// Usually, there is only one meshes. But one can split the devices in different meshes. The meshes overlap
// the concrete devices used.
//
// See details of XLA Shardy in [1]
//
// [1] https://github.com/openxla/shardy
func (b *Builder) WithShardy(meshes ...*shardy.DeviceMesh) *Builder {
	b.meshes = meshes
	b.WithNumReplicas(1)
	numDevices := 0
	for _, mesh := range meshes {
		numDevices = max(numDevices, mesh.NumDevices())
	}
	b.WithNumPartitions(numDevices)
	return b
}

// Meshes returns the meshes configured with WithShardy.
func (b *Builder) Meshes() []*shardy.DeviceMesh {
	return b.meshes
}

// NewShardingSpec creates a new ShardingSpec using the first mesh configured with WithShardy.
// It returns nil if no mesh was not configured.
//
// This is a shortcut to NewShardingSpecByMeshIx(0).
func (b *Builder) NewShardingSpec() *shardy.ShardingSpec {
	if len(b.meshes) == 0 {
		return nil
	}
	return shardy.NewShardingSpec(b.meshes[0])
}

// NewShardingSpecByMeshIx creates a new ShardingSpec for the meshIdx (the order given by WithShardy).
//
// It may return nil if meshIdx is out of range.
func (b *Builder) NewShardingSpecByMeshIx(meshIdx int) *shardy.ShardingSpec {
	if meshIdx < 0 || meshIdx >= len(b.meshes) {
		return nil
	}
	return shardy.NewShardingSpec(b.meshes[meshIdx])
}
