package types

import (
	"fmt"
	"slices"
	"strings"

	"github.com/gomlx/go-xla/pkg/types/dtypes"
)

// ComparisonType enum defined for the Compare op.
type ComparisonType int

//go:generate go tool enumer -type=ComparisonType -output=gen_comparisontype_enumer.go ops.go

const (
	// CompareFloat are used for floating point comparisons.
	CompareFloat ComparisonType = iota

	// CompareTotalOrder version of the operation enforces `-NaN < -Inf < -Finite < -0 < +0 < +Finite < +Inf < +NaN`.
	CompareTotalOrder

	CompareSigned
	CompareUnsigned
)

// ToStableHLO returns the StableHLO representation of the comparison type.
func (c ComparisonType) ToStableHLO() string {
	switch c {
	case CompareFloat:
		return "#stablehlo<comparison_type FLOAT>"
	case CompareTotalOrder:
		return "#stablehlo<comparison_type TOTALORDER>"
	case CompareSigned:
		return "#stablehlo<comparison_type SIGNED>"
	case CompareUnsigned:
		return "#stablehlo<comparison_type UNSIGNED>"
	}
	return fmt.Sprintf("#stablehlo<comparison_type UNKNOWN %d>", c)
}

// ComparisonDirection enum defined for the Compare op.
type ComparisonDirection int

//go:generate go tool enumer -type=ComparisonDirection -trimprefix=Compare -output=gen_comparisondirection_enumer.go ops.go

const (
	CompareEQ ComparisonDirection = iota
	CompareGE
	CompareGT
	CompareLE
	CompareLT
	CompareNE
)

func (c ComparisonDirection) ToStableHLO() string {
	switch c {
	case CompareEQ:
		return "#stablehlo<comparison_direction EQ>"
	case CompareLE:
		return "#stablehlo<comparison_direction LE>"
	case CompareNE:
		return "#stablehlo<comparison_direction NE>"
	case CompareLT:
		return "#stablehlo<comparison_direction LT>"
	case CompareGT:
		return "#stablehlo<comparison_direction GT>"
	case CompareGE:
		return "#stablehlo<comparison_direction GE>"
	}
	return fmt.Sprintf("#stablehlo<comparison_direction UNKNOWN %d>", c)
}

// ConvolveAxesConfig defines the interpretation of the input/kernel/output tensor axes.
// There must be the same number of spatial dimensions (axes) for each of the 3 tensors.
// Input and output have batch and channel axes. Kernel has inputChannel and outputChannel axes.
//
// See Builder.ConvGeneral
type ConvolveAxesConfig struct {
	InputBatch, InputChannels int
	InputSpatial              []int

	KernelInputChannels, KernelOutputChannels int
	KernelSpatial                             []int

	OutputBatch, OutputChannels int
	OutputSpatial               []int
}

// Clone returns a deep copy of the structure.
func (c ConvolveAxesConfig) Clone() ConvolveAxesConfig {
	var c2 ConvolveAxesConfig
	c2 = c
	c2.InputSpatial = slices.Clone(c.InputSpatial)
	c2.KernelSpatial = slices.Clone(c.KernelSpatial)
	c2.OutputSpatial = slices.Clone(c.OutputSpatial)
	return c2
}

// DotGeneralPrecisionType defines the precision of the dot product.
//
// It controls the tradeoff between speed and accuracy for computations on accelerator backends.
// This can be one of the following (at the moment, the semantics of these enum values are underspecified,
// but they are planning to address this in #755 -- https://github.com/openxla/stablehlo/issues/755):
type DotGeneralPrecisionType int

//go:generate go tool enumer -type=DotGeneralPrecisionType -trimprefix=DotGeneralPrecision -output=gen_dotgeneralprecisiontype_enumer.go ops.go

const (
	// DotGeneralPrecisionDefault is the fastest calculation, but the least accurate approximation to the original number.
	DotGeneralPrecisionDefault DotGeneralPrecisionType = iota
	DotGeneralPrecisionHigh
	DotGeneralPrecisionHighest
)

func (p DotGeneralPrecisionType) ToStableHLO() string {
	return strings.ToUpper(p.String())
}

// FloatPrecisionType defines the precision used during floating point operations.
// In particular, modern GPUs accept the TF32 type which sacrifices some accuracy for
// significant speed improvements.
type FloatPrecisionType struct {
	// TF32 is used for the TF32 precision type.
	TF32 bool

	// DType is used for non-TF32 precision types.
	// It must be a float type.
	DType dtypes.DType
}

func (f FloatPrecisionType) ToStableHLO() string {
	if f.TF32 {
		return "tf32"
	}
	return f.DType.ToStableHLO()
}

// DotGeneralAlgorithm defines fine-control of the algorithm used for the dot product.
type DotGeneralAlgorithm struct {
	// LhsPrecisionType, RhsPrecisionType that the LHS and RHS of the operation are rounded to.
	// Precision types are independent of the storage types of the inputs and the output.
	LhsPrecisionType, RhsPrecisionType FloatPrecisionType

	// AccumulationType defines the type of the accumulator used for the dot product.
	AccumulationType FloatPrecisionType

	// LhsComponentCount, RhsComponentCount and NumPrimitiveOperations apply when we are doing an algorithm which
	// decomposes the LHS and/or RHS into multiple components and does multiple "primitive" dot operations on those values -
	// usually to emulate a higher precision (e.g.: Leveraging the bfloat16 Artificial Intelligence Datatype For
	// Higher-Precision Computations: bf16_6x tf32_3x -- https://arxiv.org/pdf/1904.06376, etc).
	// For algorithms with no decomposition, these values should be set to 1
	LhsComponentCount, RhsComponentCount, NumPrimitiveOperations int

	// AllowImpreciseAccumulation to specify if accumulation in lower precision is permitted for some steps
	// (e.g. CUBLASLT_MATMUL_DESC_FAST_ACCUM).
	AllowImpreciseAccumulation bool
}

// RNGBitGeneratorAlgorithm used by the RngBitGenerator operation.
type RNGBitGeneratorAlgorithm int

const (
	RNGDefault RNGBitGeneratorAlgorithm = iota
	RNGPhilox
	RNGThreeFry
)

//go:generate go tool enumer -type=RNGBitGeneratorAlgorithm -trimprefix=RNG -output=gen_rngbitgeneratoralgorithm_enumer.go -transform=snake ops.go

// FFTType defines the type of the FFT operation, see FFT.
type FFTType int

const (
	// FFTForward - complex in, complex out.
	FFTForward FFTType = iota

	// FFTInverse - complex in, complex out.
	FFTInverse

	// FFTForwardReal - real in, fft_length / 2 + 1 complex out
	FFTForwardReal

	// FFTInverseReal - fft_length / 2 + 1 complex in
	FFTInverseReal
)

//go:generate go tool enumer -type FFTType -trimprefix=FFT -output=gen_ffttype_enumer.go ops.go

// ToStableHLO returns the StableHLO representation of the FFT type.
func (t FFTType) ToStableHLO() string {
	switch t {
	case FFTForward:
		return "FFT"
	case FFTInverse:
		return "IFFT"
	case FFTForwardReal:
		return "RFFT"
	case FFTInverseReal:
		return "IRFFT"
	default:
		return "FFT_UNKNOWN_TYPE"
	}
}

// ChannelType defines the communication dimension for a collective op.
// It is int64 to match the i64 type in the StableHLO spec.
type ChannelType int

//go:generate go tool enumer -type=ChannelType -output=gen_channeltype_enumer.go -transform=snake ops.go

const (
	// CrossReplica communicates across replicas (data parallelism).
	// This is the default.
	CrossReplica ChannelType = 0

	// CrossPartition communicates across partitions (model parallelism).
	CrossPartition ChannelType = 1
)

// CollectiveConfig provides advanced, optional configuration for collective operations.
// Pass this as the last (optional) argument to collective ops.
type CollectiveConfig struct {
	// ChannelType specifies the communication dimension.
	// Defaults to CrossReplica (0).
	ChannelType ChannelType

	// ChannelID, if non-nil, forces a specific channel ID (the 'handle').
	// If nil, a unique ID will be automatically generated.
	// This is **required** for MPMD (multi-program, multi-data) to manually link ops across programs.
	ChannelID *int

	// UseGlobalDeviceIDs changes the interpretation of replica_groups
	// from replica IDs to global device IDs.
	// This only applies to AllReduce, not CollectiveBroadcast.
	// Defaults to false.
	UseGlobalDeviceIDs bool
}
