package shardy

import (
	"testing"
)

func TestShardSpec_ToStableHLO(t *testing.T) {
	mesh, err := NewDeviceMesh("test_mesh", []int{4, 2}, []string{"z", "a"})
	if err != nil {
		t.Fatalf("NewDeviceMesh() error = %v", err)
	}
	testCases := []struct {
		name     string
		spec     *ShardingSpec
		expected string
	}{
		{
			name:     "Replicated",
			spec:     NewShardingSpec(mesh).AddReplicated(),
			expected: "#sdy.sharding<@test_mesh, [{}], replicated={a, z}>",
		},
		{
			name:     "Sharded",
			spec:     NewShardingSpec(mesh).AddShardedAxis("z"),
			expected: "#sdy.sharding<@test_mesh, [{z}], replicated={a}>",
		},
		{
			name:     "Sharded with multiple axes",
			spec:     NewShardingSpec(mesh).AddShardedAxis("z", "a"),
			expected: "#sdy.sharding<@test_mesh, [{z, a}]>",
		},
		{
			name: "Sharded with sub-axis",
			spec: &ShardingSpec{
				Mesh: mesh,
				Axes: []TensorAxisSpec{
					{MeshAxes: []MeshAxisSpec{{AxisName: "a", PreSize: 1, Size: 2}}},
				},
			},
			expected: "#sdy.sharding<@test_mesh, [{a:(1)2}], replicated={z}>",
		},
		{
			name:     "Opened",
			spec:     &ShardingSpec{Mesh: mesh, Axes: []TensorAxisSpec{{Opened: true}}},
			expected: "#sdy.sharding<@test_mesh, [{?}], replicated={a, z}>",
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			if got := tc.spec.ToStableHLO(); got != tc.expected {
				t.Errorf("ToStableHLO() = %q, want %q", got, tc.expected)
			}
		})
	}
}

func TestShardSpec_Validate(t *testing.T) {
	mesh, err := NewDeviceMesh("test_mesh", []int{2, 8}, []string{"z", "a"})
	if err != nil {
		t.Fatalf("NewDeviceMesh() error = %v", err)
	}
	testCases := []struct {
		name        string
		spec        *ShardingSpec
		expectError bool
	}{
		{
			name:        "Valid sharding",
			spec:        NewShardingSpec(mesh).AddShardedAxis("z"),
			expectError: false,
		},
		{
			name:        "Unknown mesh axis",
			spec:        NewShardingSpec(mesh).AddShardedAxis("x"),
			expectError: true,
		},
		{
			name: "Valid sub-axis",
			spec: &ShardingSpec{
				Mesh: mesh,
				Axes: []TensorAxisSpec{
					{MeshAxes: []MeshAxisSpec{{AxisName: "a", PreSize: 2, Size: 4}}},
				},
			},
			expectError: false,
		},
		{
			name: "Invalid sub-axis (PreSize)",
			spec: &ShardingSpec{
				Mesh: mesh,
				Axes: []TensorAxisSpec{
					{MeshAxes: []MeshAxisSpec{{AxisName: "a", PreSize: 0, Size: 4}}},
				},
			},
			expectError: true,
		},
		{
			name: "Invalid sub-axis (Size)",
			spec: &ShardingSpec{
				Mesh: mesh,
				Axes: []TensorAxisSpec{
					{MeshAxes: []MeshAxisSpec{{AxisName: "a", PreSize: 2, Size: 5}}},
				},
			},
			expectError: true,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			err := tc.spec.Validate()
			if tc.expectError {
				if err == nil {
					t.Error("Validate() expected error, got nil")
				}
			} else {
				if err != nil {
					t.Errorf("Validate() error = %v", err)
				}
			}
		})
	}
}
