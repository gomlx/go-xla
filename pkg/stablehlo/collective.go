package stablehlo

import (
	"fmt"
	"strings"

	"github.com/gomlx/go-xla/internal/optypes"
	"github.com/gomlx/go-xla/internal/shapeinference"
	"github.com/gomlx/go-xla/pkg/types"
	"github.com/pkg/errors"
)

// formatReplicaGroups converts a 2D Go slice into the StableHLO dense tensor literal format.
// Example: [[0, 1], [2, 3]] -> "dense<[[0, 1], [2, 3]]> : tensor<2x2xi64>"
func formatReplicaGroups(groups [][]int) literalStr {
	if len(groups) == 0 {
		return "dense<[]> : tensor<0x0xi64>"
	}

	var sb strings.Builder
	sb.WriteString("dense<[")
	for i, group := range groups {
		if i > 0 {
			sb.WriteString(", ")
		}
		sb.WriteString("[")
		for j, replica := range group {
			if j > 0 {
				sb.WriteString(", ")
			}
			sb.WriteString(fmt.Sprintf("%d", replica))
		}
		sb.WriteString("]")
	}
	sb.WriteString("]>")
	sb.WriteString(fmt.Sprintf(" : tensor<%dx%dxi64>", len(groups), len(groups[0])))
	return literalStr(sb.String())
}

// CollectiveBroadcast broadcasts the value from the first replica (in each group) to all others.
// The returned shape is the same as the source.
// Devices not included in any replica group will return zeros as their output (with the same shape as the input).
//
//   - operand: The tensor to be broadcasted. In an SPMD setup, this op will be called on all
//     replicas, but only the operand from the source device (the first device in
//     the replica_group) will be used.
//   - replicaGroups: A 2D array defining the communicating device groups. For standard data
//     parallelism, this is typically a single group with all the replica numbers --
//     notice it's not the device numbers by the replica numbers (there is an indirection).
//     Except if the config sets UseGlobalDeviceIDs, in which case they are interpreted as device
//     numbers. E.g., `[[0, 1, 2, 3]]`.
//   - config: Optional configuration of the channels to be used. This is shouldn't be used for SPMD programs.
//
// Consider using Builder.WithShardy for distributed computation instead: other forms of distributed
// (collective) computation across devices are not tested and may not work.
func CollectiveBroadcast(operand *Value, replicaGroups [][]int, config ...*types.CollectiveConfig) (*Value, error) {
	op := optypes.CollectiveBroadcast
	fn := operand.fn
	if fn.Returned {
		return nil, errors.Errorf("cannot add operation %s after returning, in function %q",
			op, fn.Name)
	}

	outputShape, err := shapeinference.CollectiveBroadcast(operand.shape, replicaGroups)
	if err != nil {
		return nil, err
	}

	var cfg *types.CollectiveConfig
	if len(config) > 1 {
		return nil, errors.Errorf("only one config can be provided, got %d", len(config))
	} else if len(config) == 1 {
		cfg = config[0]
	}

	if cfg != nil && (cfg.UseGlobalDeviceIDs || cfg.ChannelType == types.CrossPartition) {
		return nil, errors.Errorf("UseGlobalDeviceIDs or CrossPartition type is not supported for CollectiveBroadcast")
	}

	stmt := fn.addOp(op, outputShape, operand)
	stmt.Attributes = map[string]any{
		"replica_groups": formatReplicaGroups(replicaGroups),
	}
	if cfg != nil {
		stmt.Attributes["channel_handle"] = fn.Builder.getChannelHandle(cfg)
	}
	return stmt.Outputs[0], nil
}

// AllReduce performs a distributed reduce operation across replicas.
// It is a distributed version of Reduce.
//
//   - operands: The tensors from the *local* replica to be reduced.
//   - replicaGroups: A 2D array defining the communicating device groups, e.g., `[[0, 1, 2, 3]]`.
//   - computation: A closure function that defines the reduction operation (e.g., SUM). It must
//     take two scalar inputs for each operand's dtype and return one scalar output of the same dtype.
//   - replicaGroups: A 2D array defining the communicating device groups. For standard data
//     parallelism, this is typically a single group with all the replica numbers --
//     notice it's not the device numbers by the replica numbers (there is an indirection).
//     Except if the config sets UseGlobalDeviceIDs, in which case they are interpreted as device
//     numbers. E.g., `[[0, 1, 2, 3]]`.
//   - config: Optional configuration of the channels to be used. This is not needed for SPMD programs.
//
// Consider using Builder.WithShardy for distributed computation instead: other forms of distributed
// (collective) computation across devices are not tested and may not work.
func AllReduce(operands []*Value, replicaGroups [][]int, computation *Function, config ...*types.CollectiveConfig) (
	[]*Value, error) {
	op := optypes.AllReduce
	if len(operands) == 0 {
		return nil, errors.Errorf("AllReduce requires at least one operand")
	}
	fn, err := innerMostFunction(operands...)
	if err != nil {
		return nil, err
	}
	if fn.Returned {
		return nil, errors.Errorf("cannot add operation %s after returning, in function %q",
			op, fn.Name)
	}
	if computation.Parent != fn {
		return nil, errors.Errorf(
			"cannot add operation %s because computation is not a StableHLO closure of %s",
			op, fn.Name)
	}

	outputShapes, err := shapeinference.AllReduce(
		valuesToShapes(operands),
		valuesToShapes(computation.Inputs),
		valuesToShapes(computation.Outputs),
		replicaGroups)
	if err != nil {
		return nil, err
	}

	var cfg *types.CollectiveConfig
	if len(config) > 1 {
		return nil, errors.Errorf("only one config can be provided, got %d", len(config))
	} else if len(config) == 1 {
		cfg = config[0]
	}

	stmt := fn.addMultiOp(op, outputShapes, operands)
	stmt.Attributes = map[string]any{
		"replica_groups": formatReplicaGroups(replicaGroups),
	}
	if cfg != nil {
		stmt.Attributes["channel_handle"] = fn.Builder.getChannelHandle(cfg)
	}
	if cfg != nil && cfg.UseGlobalDeviceIDs {
		stmt.Attributes["use_global_device_ids"] = true
	}
	stmt.AddFunctionParameter("computation", computation)
	return stmt.Outputs, nil
}

// AllGather concatenates the operand from each replica along a specified dimension.
//
//   - operand: The tensor from the *local* replica to be gathered.
//   - replicaGroups: A 2D array defining the communicating device groups.
//   - allGatherDim: The dimension along which to concatenate the operands.
//   - config: Optional configuration of the channels to be used.
//
// Consider using Builder.WithShardy for distributed computation instead: other forms of distributed
// (collective) computation across devices are not tested and may not work.
func AllGather(operand *Value, replicaGroups [][]int, allGatherDim int, config ...*types.CollectiveConfig) (*Value, error) {
	op := optypes.AllGather
	fn := operand.fn
	if fn.Returned {
		return nil, errors.Errorf("cannot add operation %s after returning, in function %q", op, fn.Name)
	}

	outputShape, err := shapeinference.AllGather(operand.shape, replicaGroups, allGatherDim)
	if err != nil {
		return nil, err
	}

	var cfg *types.CollectiveConfig
	if len(config) > 1 {
		return nil, errors.Errorf("only one config can be provided, got %d", len(config))
	} else if len(config) == 1 {
		cfg = config[0]
	}

	stmt := fn.addOp(op, outputShape, operand)
	stmt.Attributes = map[string]any{
		"replica_groups": formatReplicaGroups(replicaGroups),
		"all_gather_dim": int64(allGatherDim),
	}
	if cfg != nil {
		stmt.Attributes["channel_handle"] = fn.Builder.getChannelHandle(cfg)
	}
	if cfg != nil && cfg.UseGlobalDeviceIDs {
		stmt.Attributes["use_global_device_ids"] = true
	}
	return stmt.Outputs[0], nil
}

// AllToAll splits the operand along a specified dimension and scatters the chunks to all replicas,
// where they are concatenated back together.
//
//   - operand: The tensor from the *local* replica.
//   - replicaGroups: A 2D array defining the communicating device groups.
//   - splitDimension: The dimension along which to split the operand.
//   - concatDimension: The dimension along which to concatenate the received chunks.
//   - splitCount: The number of chunks to split the operand into. This must match the size of the replica groups.
//   - config: Optional configuration of the channels to be used.
//
// Consider using Builder.WithShardy for distributed computation instead: other forms of distributed
// (collective) computation across devices are not tested and may not work.
func AllToAll(operand *Value, replicaGroups [][]int, splitDimension, concatDimension, splitCount int, config ...*types.CollectiveConfig) (*Value, error) {
	op := optypes.AllToAll
	fn := operand.fn
	if fn.Returned {
		return nil, errors.Errorf("cannot add operation %s after returning, in function %q", op, fn.Name)
	}

	outputShape, err := shapeinference.AllToAll(operand.shape, replicaGroups, splitDimension, concatDimension, splitCount)
	if err != nil {
		return nil, err
	}

	var cfg *types.CollectiveConfig
	if len(config) > 1 {
		return nil, errors.Errorf("only one config can be provided, got %d", len(config))
	} else if len(config) == 1 {
		cfg = config[0]
	}

	stmt := fn.addOp(op, outputShape, operand)
	stmt.Attributes = map[string]any{
		"replica_groups":   formatReplicaGroups(replicaGroups),
		"split_dimension":  int64(splitDimension),
		"concat_dimension": int64(concatDimension),
		"split_count":      int64(splitCount),
	}
	if cfg != nil {
		stmt.Attributes["channel_handle"] = fn.Builder.getChannelHandle(cfg)
	}
	if cfg != nil && cfg.UseGlobalDeviceIDs {
		stmt.Attributes["use_global_device_ids"] = true
	}
	return stmt.Outputs[0], nil
}

// formatSourceTargetPairs converts a 2D Go slice into the StableHLO dense tensor literal format.
// Example: [[0, 1], [2, 3]] -> "dense<[[0, 1], [2, 3]]> : tensor<2x2xi64>"
func formatSourceTargetPairs(pairs [][2]int) literalStr {
	if len(pairs) == 0 {
		return "dense<[]> : tensor<0x2xi64>"
	}

	var sb strings.Builder
	sb.WriteString("dense<[")
	for i, pair := range pairs {
		if i > 0 {
			sb.WriteString(", ")
		}
		sb.WriteString(fmt.Sprintf("[%d, %d]", pair[0], pair[1]))
	}
	sb.WriteString("]>")
	sb.WriteString(fmt.Sprintf(" : tensor<%dx2xi64>", len(pairs)))
	return literalStr(sb.String())
}

// CollectivePermute sends the operand from a source replica to a target replica.
//
//   - operand: The tensor from the *local* replica.
//   - sourceTargetPairs: A 2D array where each inner array is a `[source, target]` pair of replica IDs.
//   - config: Optional configuration of the channels to be used.
//
// Consider using Builder.WithShardy for distributed computation instead: other forms of distributed
// (collective) computation across devices are not tested and may not work.
func CollectivePermute(operand *Value, sourceTargetPairs [][2]int, config ...*types.CollectiveConfig) (*Value, error) {
	op := optypes.CollectivePermute
	fn := operand.fn
	if fn.Returned {
		return nil, errors.Errorf("cannot add operation %s after returning, in function %q", op, fn.Name)
	}

	outputShape, err := shapeinference.CollectivePermute(operand.shape, sourceTargetPairs)
	if err != nil {
		return nil, err
	}

	var cfg *types.CollectiveConfig
	if len(config) > 1 {
		return nil, errors.Errorf("only one config can be provided, got %d", len(config))
	} else if len(config) == 1 {
		cfg = config[0]
	}

	stmt := fn.addOp(op, outputShape, operand)
	stmt.Attributes = map[string]any{
		"source_target_pairs": formatSourceTargetPairs(sourceTargetPairs),
	}
	if cfg != nil {
		stmt.Attributes["channel_handle"] = fn.Builder.getChannelHandle(cfg)
	}
	if cfg != nil && cfg.UseGlobalDeviceIDs {
		stmt.Attributes["use_global_device_ids"] = true
	}
	return stmt.Outputs[0], nil
}
