package shardy

import (
	"fmt"
	"sort"
	"strings"

	"github.com/gomlx/go-xla/pkg/types/shapes"
	"github.com/pkg/errors"
)

// ShardingSpec (also known as PartitionSpec in JAX) defines how a logical tensor is to be sharded (partitioned) across
// a DeviceMesh. This is used by Shardy, and is based on its documentation in [1].
//
// The definition is per axis of the logical tensor -- and not per axis of the Mesh, a common confusion.
// If not all axes of the Tensor are defined, the tail axes are considered simply to be replicated across the whole
// mesh.
//
// Each tensor axis can be replicated or sharded across one or more mesh axes.
//
// Example:
//
//	mesh := NewDeviceMesh("my_mesh", []int{2, 2}, []string{"data", "model"})
//
//	// Input's "batch" axis is sharded across the "data" axis of the mesh.
//	inputSharding := NewShardingSpec(mesh).AddShardedAxis("data")
//
//	// First axis is replicated, second is shared across "model" devices
//	variableSharding := NewShardingSpec(mesh).AddReplicated().AddShardedAxis("model")
//
//	// Second axis is sharded across both "data" and "model" devices.
//	 largeWeights := NewShardingSpec(mesh).AddReplicated().AddShardedAxis("data", "model")
//
// There are two advanced features supported but not tested (pls if you need let us know how it goes, or if you find
// any issues):
//
//  1. The tensor can also be sharded across mesh "sub-axes" -- seed detailed documentation in [1]
//  2. If using ShardingSpec for hints, instead of mesh axes one can give an "open" (in StableHLO marked as "?")
//     axis, with the semantics that XLA Shardy can choose any mesh axis (or axes) to shard the tensor. See [1].
//
// [1] https://github.com/openxla/shardy/blob/main/docs/sharding_representation.md
type ShardingSpec struct {
	Mesh *DeviceMesh
	Axes []TensorAxisSpec
}

// TensorAxisSpec specifies how a tensor axis is to be sharded (or replicated).
// See details in ShardingSpec.
//
// Usually, one would create this using ShardingSpec.AddShardedAxis or ShardingSpec.AddReplicated
type TensorAxisSpec struct {
	MeshAxes []MeshAxisSpec
	Opened   bool // If opened to further sharding.
}

type MeshAxisSpec struct {
	AxisName string

	// PreSize, Size are only set if defining a sub-axis of the mesh.
	PreSize, Size int
}

// NewShardingSpec creates a new ShardingSpec.
func NewShardingSpec(mesh *DeviceMesh) *ShardingSpec {
	return &ShardingSpec{mesh, make([]TensorAxisSpec, 0)}
}

// AddShardedAxis adds a new sharded axis to the ShardingSpec using one or more mesh axes.
//
// It returns itself, so calls can be chained.
func (s *ShardingSpec) AddShardedAxis(meshAxesNames ...string) *ShardingSpec {
	axisSpec := TensorAxisSpec{}
	for _, meshAxisName := range meshAxesNames {
		axisSpec.MeshAxes = append(axisSpec.MeshAxes, MeshAxisSpec{AxisName: meshAxisName})
	}
	s.Axes = append(s.Axes, axisSpec)
	return s
}

// AddReplicated adds a new replicated axis to the ShardingSpec.
//
// It returns itself, so calls can be chained.
func (s *ShardingSpec) AddReplicated() *ShardingSpec {
	s.Axes = append(s.Axes, TensorAxisSpec{})
	return s
}

// Rank returns the number of axes this ShardingSpec describes.
//
// Notice this may be smaller than the rank of the tensor using it: if a tensor axis is not defined in ShardingSpec,
// it is assumed to be replicated.
func (s *ShardingSpec) Rank() int {
	return len(s.Axes)
}

// IsReplicated returns true if the tensor is fully replicated
// (i.e., not sharded along any axis and not marked as "open").
func (s *ShardingSpec) IsReplicated() bool {
	for _, axisSpec := range s.Axes {
		if axisSpec.MeshAxes != nil || axisSpec.Opened {
			return false
		}
	}
	return true
}

// Validate checks that the ShardingSpec is valid for the given mesh.
func (s *ShardingSpec) Validate() error {
	for i, axisSpec := range s.Axes {
		for j, meshAxisSpec := range axisSpec.MeshAxes {
			axisName := meshAxisSpec.AxisName
			if axisName == "" {
				return errors.Errorf(
					"ShardingSpec tensor axis %d, mesh axis #%d refers to empty mesh axis name", i, j)
			}
			axisIdx, ok := s.Mesh.nameToAxis[axisName]
			if !ok {
				return errors.Errorf(
					"ShardingSpec tensor axis %d, mesh axis #%d refers to unknown mesh axis %q",
					i, j, axisName)
			}
			meshAxisSize := s.Mesh.axesSizes[axisIdx]

			// Check sub-axis specification.
			if meshAxisSpec.Size > 0 {
				if meshAxisSpec.PreSize <= 0 {
					return errors.Errorf("ShardingSpec tensor axis %d, mesh axis #%d %q has invalid PreSize %d",
						i, j, axisName, meshAxisSpec.PreSize)
				}
				if meshAxisSize%(meshAxisSpec.PreSize*meshAxisSpec.Size) != 0 {
					return errors.Errorf(
						"ShardingSpec tensor axis %d, mesh axis #%d %q with PreSize %d and Size %d is not "+
							"compatible with mesh axis of size %d",
						i, j, axisName, meshAxisSpec.PreSize, meshAxisSpec.Size, meshAxisSize)
				}
			}
		}
	}
	return nil
}

func (s *ShardingSpec) ValidateShape(shape shapes.Shape) error {
	if s == nil {
		// No sharding spec (nil) means fully replicated, and it's always valid for any shape.
		return nil
	}
	err := s.Validate()
	if err != nil {
		return err
	}
	if s.Rank() > shape.Rank() {
		return errors.Errorf("ShardingSpec shape rank %d is larger than tensor rank %d", s.Rank(), shape.Rank())
	}
	return nil
}

// ToStableHLO converts the ShardingSpec to its StableHLO string representation.
// See details in:
// https://github.com/openxla/shardy/blob/main/docs/sharding_representation.md
func (s *ShardingSpec) ToStableHLO() string {
	var dimShardings []string
	replicatedAxes := make(map[string]bool)
	for _, axisName := range s.Mesh.axesNames {
		replicatedAxes[axisName] = true
	}

	for _, axisSpec := range s.Axes {
		var hloAxes []string
		for _, meshAxisSpec := range axisSpec.MeshAxes {
			delete(replicatedAxes, meshAxisSpec.AxisName)
			if meshAxisSpec.Size > 0 {
				hloAxes = append(hloAxes, fmt.Sprintf("%s:(%d)%d", meshAxisSpec.AxisName, meshAxisSpec.PreSize, meshAxisSpec.Size))
			} else {
				hloAxes = append(hloAxes, meshAxisSpec.AxisName)
			}
		}
		if axisSpec.Opened {
			hloAxes = append(hloAxes, "?")
		}
		dimShardings = append(dimShardings, fmt.Sprintf("{%s}", strings.Join(hloAxes, ", ")))
	}

	var replicatedStrs []string
	for axisName := range replicatedAxes {
		replicatedStrs = append(replicatedStrs, axisName)
	}
	sort.Strings(replicatedStrs)

	replicatedPart := ""
	if len(replicatedStrs) > 0 {
		replicatedPart = fmt.Sprintf(", replicated={%s}", strings.Join(replicatedStrs, ", "))
	}
	return fmt.Sprintf("#sdy.sharding<@%s, [%s]%s>", s.Mesh.Name(), strings.Join(dimShardings, ", "), replicatedPart)
}

// ToValueAttribute converts the ShardingSpec to a StableHLO attribute of a value with the given shape.
//
// Notice the rank of the ShardingSpec may be smaller than the rank of the shape, in which case the extra axes are
// assumed to be replicated (empty).
//
// E.g.: "#sdy.sharding<@mesh, [{\"data\"}, {}]>"
func (s *ShardingSpec) ToValueAttribute(shape shapes.Shape) string {
	var buf strings.Builder
	w := func(format string, args ...any) {
		buf.WriteString(fmt.Sprintf(format, args...))
	}
	w("#sdy.sharding<@%s, [", s.Mesh.Name())
	for axisIdx := range shape.Rank() {
		if axisIdx > 0 {
			w(", ")
		}
		if axisIdx >= len(s.Axes) {
			w("{}")
			continue
		}
		tensorAxisSpec := s.Axes[axisIdx]
		if len(tensorAxisSpec.MeshAxes) == 0 {
			if tensorAxisSpec.Opened {
				w("{?}")
			} else {
				w("{}")
			}
			continue
		}
		w("{")
		for meshAxisIdx, meshAxisSpec := range tensorAxisSpec.MeshAxes {
			if meshAxisIdx > 0 {
				w(", ")
			}
			if meshAxisSpec.Size > 0 {
				w("\"%s\":(%d)%d", meshAxisSpec.AxisName, meshAxisSpec.PreSize, meshAxisSpec.Size)
			} else {
				w("\"%s\"", meshAxisSpec.AxisName)
			}
		}
		if tensorAxisSpec.Opened {
			w(", ?")
		}
		w("}")
	}
	w("]>")
	return buf.String()
}
