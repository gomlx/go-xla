// Package optypes defines OpType and lists the supported operations.
package optypes

import (
	"fmt"

	"github.com/gomlx/go-xla/internal/utils"
)

// OpType is an enum of all generic operations that ToStableHLO can support -- not all are implemented yet.
type OpType int

//go:generate go tool enumer -type=OpType -output=gen_optype_enumer.go optypes.go

const (
	Invalid OpType = iota
	FuncReturn
	Constant
	Identity

	Abs
	Add
	AllGather
	AllReduce
	AllToAll
	And
	Atan2
	BatchNormInference
	BatchNormTraining
	BatchNormGrad
	BitcastConvert
	BroadcastInDim
	Call
	Cbrt
	Ceil
	Clamp
	CollectiveBroadcast
	CollectivePermute
	Compare
	Complex
	Concatenate
	Convert
	Convolution
	Cosine
	CountLeadingZeros
	Divide
	DotGeneral
	DynamicBroadcastInDim
	DynamicConv
	DynamicGather
	DynamicIota
	DynamicPad
	DynamicSlice
	DynamicUpdateSlice
	Erf
	Exponential
	ExponentialMinusOne
	Fft
	Floor
	Gather
	If
	Imag
	IsFinite
	Iota
	Log
	LogPlusOne
	Logistic
	Maximum
	Minimum
	Multiply
	Negate
	Not
	OptimizationBarrier
	Or
	Pad
	Popcnt
	Power
	Real
	Remainder
	Reduce
	ReduceWindow
	Reshape
	Reverse
	RNGBitGenerator
	RoundNearestAfz
	RoundNearestEven
	Rsqrt
	Scatter
	Select
	SelectAndScatter
	ShiftLeft
	ShiftRightArithmetic
	ShiftRightLogical
	Sign
	Sine
	Slice
	Sort
	Sqrt
	Subtract
	Tan
	Tanh
	Transpose
	UniformDequantize
	UniformQuantize
	While
	Xor

	GetDimensionSize

	// Here the ones not implemented yet, please add an issue in the repo if you need them.

	Case
	Cholesky
	Composite
	CustomCall
	DynamicReshape
	GetTupleElement
	Infeed
	Outfeed
	PartitionId
	Recv
	ReducePrecision
	ReduceScatter
	Send
	TriangularSolve
	Tuple

	// Last should always be kept the last, it is used as a counter/marker for .
	Last
)

var (
	// stableHLOMappings maps OpType to the corresponding StableHLO name, when the default
	// "snake case" doesn't work.
	stableHLOMappings = map[OpType]string{
		FuncReturn: "stablehlo.return",
		Call:       "func.call",
		Erf:        "chlo.erf",
		AllReduce:  "stablehlo.all_reduce"}
)

// ToStableHLO returns the ToStableHLO name of the operation.
func (op OpType) ToStableHLO() string {
	name, ok := stableHLOMappings[op]
	if !ok {
		name = fmt.Sprintf("stablehlo.%s", utils.ToSnakeCase(op.String()))
	}
	return name
}
