package shardy_test

import (
	"reflect"
	"strings"
	"testing"

	"github.com/gomlx/go-xla/pkg/types/shardy"
)

func TestDeviceMesh(t *testing.T) {
	t.Run("NewDeviceMesh_Valid", func(t *testing.T) {
		tests := []struct {
			name          string
			shape         []int
			axisNames     []string
			wantRank      int
			wantNum       int
			wantStableHLO string
		}{
			{
				name:          "1D mesh",
				shape:         []int{8},
				axisNames:     []string{"replica"},
				wantRank:      1,
				wantNum:       8,
				wantStableHLO: `sdy.mesh @mesh = <["replica"=8]>`,
			},
			{
				name:          "2D mesh",
				shape:         []int{2, 4},
				axisNames:     []string{"x", "y"},
				wantRank:      2,
				wantNum:       8,
				wantStableHLO: `sdy.mesh @mesh = <["x"=2, "y"=4]>`,
			},
			{
				name:          "3D mesh",
				shape:         []int{2, 2, 2},
				axisNames:     []string{"x", "y", "z"},
				wantRank:      3,
				wantNum:       8,
				wantStableHLO: `sdy.mesh @mesh = <["x"=2, "y"=2, "z"=2]>`,
			},
			{
				name:          "single device",
				shape:         []int{1},
				axisNames:     []string{"replica"},
				wantRank:      1,
				wantNum:       1,
				wantStableHLO: `sdy.mesh @mesh = <["replica"=1]>`,
			},
		}

		for _, tt := range tests {
			t.Run(tt.name, func(t *testing.T) {
				mesh, err := shardy.NewDeviceMesh("mesh", tt.shape, tt.axisNames)
				if err != nil {
					t.Fatalf("NewDeviceMesh() error = %v", err)
				}
				if mesh == nil {
					t.Fatal("NewDeviceMesh() returned nil")
				}
				if got := mesh.Rank(); got != tt.wantRank {
					t.Errorf("Rank() = %d, want %d", got, tt.wantRank)
				}
				if got := mesh.NumDevices(); got != tt.wantNum {
					t.Errorf("NumDevices() = %d, want %d", got, tt.wantNum)
				}
				if got := mesh.ToStableHLO(); got != tt.wantStableHLO {
					t.Errorf("ToStableHLO() = %q, want %q", got, tt.wantStableHLO)
				}
			})
		}
	})

	t.Run("NewDeviceMesh_Errors", func(t *testing.T) {
		tests := []struct {
			name      string
			shape     []int
			axisNames []string
			wantErr   string
		}{
			{
				name:      "mismatched lengths",
				shape:     []int{2, 4},
				axisNames: []string{"x"},
				wantErr:   "axesSizes and axesNames must have the same length",
			},
			{
				name:      "empty axesSizes",
				shape:     []int{},
				axisNames: []string{},
				wantErr:   "DeviceMesh axesSizes cannot be empty",
			},
			{
				name:      "empty axis name",
				shape:     []int{4},
				axisNames: []string{""},
				wantErr:   "axis name at index 0 cannot be empty",
			},
			{
				name:      "duplicate axis names",
				shape:     []int{2, 4},
				axisNames: []string{"x", "x"},
				wantErr:   "axis name \"x\" is duplicated",
			},
		}

		for _, tt := range tests {
			t.Run(tt.name, func(t *testing.T) {
				mesh, err := shardy.NewDeviceMesh("mesh", tt.shape, tt.axisNames)
				if err == nil {
					t.Error("NewDeviceMesh() expected error, got nil")
				}
				if mesh != nil {
					t.Error("NewDeviceMesh() expected nil mesh on error")
				}
				if err != nil && !strings.Contains(err.Error(), tt.wantErr) {
					t.Errorf("NewDeviceMesh() error = %q, want substring %q", err.Error(), tt.wantErr)
				}
			})
		}
	})

	t.Run("AxesNames", func(t *testing.T) {
		mesh, err := shardy.NewDeviceMesh("mesh", []int{2, 4}, []string{"x", "y"})
		if err != nil {
			t.Fatalf("NewDeviceMesh() error = %v", err)
		}

		axisNames := mesh.AxesNames()
		if !reflect.DeepEqual(axisNames, []string{"x", "y"}) {
			t.Errorf("AxesNames() = %v, want %v", axisNames, []string{"x", "y"})
		}

		// Verify it returns a copy
		axisNames[0] = "modified"
		if !reflect.DeepEqual(mesh.AxesNames(), []string{"x", "y"}) {
			t.Errorf("AxesNames() modified original, want %v", []string{"x", "y"})
		}
	})

	t.Run("Shape", func(t *testing.T) {
		mesh, err := shardy.NewDeviceMesh("mesh", []int{2, 4}, []string{"x", "y"})
		if err != nil {
			t.Fatalf("NewDeviceMesh() error = %v", err)
		}

		axesSizes := mesh.AxesSizes()
		if !reflect.DeepEqual(axesSizes, []int{2, 4}) {
			t.Errorf("AxesSizes() = %v, want %v", axesSizes, []int{2, 4})
		}

		// Verify it returns a copy
		axesSizes[0] = 99
		if !reflect.DeepEqual(mesh.AxesSizes(), []int{2, 4}) {
			t.Errorf("AxesSizes() modified original, want %v", []int{2, 4})
		}
	})

	t.Run("AxisSize", func(t *testing.T) {
		mesh, err := shardy.NewDeviceMesh("mesh", []int{2, 4}, []string{"x", "y"})
		if err != nil {
			t.Fatalf("NewDeviceMesh() error = %v", err)
		}

		tests := []struct {
			name     string
			axisName string
			wantSize int
			wantErr  bool
		}{
			{
				name:     "valid axis x",
				axisName: "x",
				wantSize: 2,
				wantErr:  false,
			},
			{
				name:     "valid axis y",
				axisName: "y",
				wantSize: 4,
				wantErr:  false,
			},
			{
				name:     "non-existent axis",
				axisName: "z",
				wantSize: 0,
				wantErr:  true,
			},
		}

		for _, tt := range tests {
			t.Run(tt.name, func(t *testing.T) {
				size, err := mesh.AxisSize(tt.axisName)
				if tt.wantErr {
					if err == nil {
						t.Error("AxisSize() expected error, got nil")
					}
					if err != nil && !strings.Contains(err.Error(), "not found") {
						t.Errorf("AxisSize() error = %q, want substring %q", err.Error(), "not found")
					}
				} else {
					if err != nil {
						t.Errorf("AxisSize() error = %v", err)
					}
					if size != tt.wantSize {
						t.Errorf("AxisSize() = %d, want %d", size, tt.wantSize)
					}
				}
			})
		}
	})

	t.Run("String", func(t *testing.T) {
		tests := []struct {
			name      string
			shape     []int
			axisNames []string
			want      string
		}{
			{
				name:      "1D mesh",
				shape:     []int{8},
				axisNames: []string{"replica"},
				want:      "DeviceMesh(axesSizes={replica: 8})",
			},
			{
				name:      "2D mesh",
				shape:     []int{2, 4},
				axisNames: []string{"x", "y"},
				want:      "DeviceMesh(axesSizes={x: 2, y: 4})",
			},
		}

		for _, tt := range tests {
			t.Run(tt.name, func(t *testing.T) {
				mesh, err := shardy.NewDeviceMesh("mesh", tt.shape, tt.axisNames)
				if err != nil {
					t.Fatalf("NewDeviceMesh() error = %v", err)
				}
				if got := mesh.String(); got != tt.want {
					t.Errorf("String() = %q, want %q", got, tt.want)
				}
			})
		}
	})

	t.Run("SetDeviceAssignment_Valid", func(t *testing.T) {
		mesh, err := shardy.NewDeviceMesh("mesh", []int{4}, []string{"replica"})
		if err != nil {
			t.Fatalf("NewDeviceMesh() error = %v", err)
		}

		tests := []struct {
			name    string
			devices []int
		}{
			{
				name:    "sequential mapping",
				devices: []int{0, 1, 2, 3},
			},
			{
				name:    "reverse mapping",
				devices: []int{3, 2, 1, 0},
			},
			{
				name:    "custom mapping",
				devices: []int{2, 1, 3, 0},
			},
		}

		for _, tt := range tests {
			t.Run(tt.name, func(t *testing.T) {
				err := mesh.SetLogicalDeviceAssignment(tt.devices...)
				if err != nil {
					t.Errorf("SetLogicalDeviceAssignment() error = %v", err)
				}
			})
		}
	})

	t.Run("SetDeviceAssignment_Errors", func(t *testing.T) {
		mesh, err := shardy.NewDeviceMesh("mesh", []int{4}, []string{"replica"})
		if err != nil {
			t.Fatalf("NewDeviceMesh() error = %v", err)
		}

		tests := []struct {
			name    string
			devices []int
			wantErr string
		}{
			{
				name:    "wrong number of devices",
				devices: []int{0, 1, 2},
				wantErr: "devices must have 4 elements",
			},
			{
				name:    "duplicate device",
				devices: []int{0, 1, 1, 3},
				wantErr: "physical device #1 is duplicated",
			},
			{
				name:    "device out of range (negative)",
				devices: []int{0, 1, -1, 3},
				wantErr: "devices must be between 0 and 3",
			},
		}

		for _, tt := range tests {
			t.Run(tt.name, func(t *testing.T) {
				err := mesh.SetLogicalDeviceAssignment(tt.devices...)
				if err == nil {
					t.Error("SetLogicalDeviceAssignment() expected error, got nil")
				}
				if err != nil && !strings.Contains(err.Error(), tt.wantErr) {
					t.Errorf("SetLogicalDeviceAssignment() error = %q, want substring %q", err.Error(), tt.wantErr)
				}
			})
		}
	})

	t.Run("DeviceToMesh_2D", func(t *testing.T) {
		mesh, err := shardy.NewDeviceMesh("mesh", []int{2, 4}, []string{"x", "y"})
		if err != nil {
			t.Fatalf("NewDeviceMesh() error = %v", err)
		}
		if got := mesh.NumDevices(); got != 8 {
			t.Errorf("NumDevices() = %d, want 8", got)
		}
	})

	t.Run("DeviceToMesh_3D", func(t *testing.T) {
		mesh, err := shardy.NewDeviceMesh("mesh", []int{2, 2, 2}, []string{"x", "y", "z"})
		if err != nil {
			t.Fatalf("NewDeviceMesh() error = %v", err)
		}
		if got := mesh.NumDevices(); got != 8 {
			t.Errorf("NumDevices() = %d, want 8", got)
		}
	})

	t.Run("DeviceToMesh_WithCustomMapping", func(t *testing.T) {
		mesh, err := shardy.NewDeviceMesh("mesh", []int{4}, []string{"replica"})
		if err != nil {
			t.Fatalf("NewDeviceMesh() error = %v", err)
		}
		err = mesh.SetLogicalDeviceAssignment(3, 2, 1, 0)
		if err != nil {
			t.Fatalf("SetLogicalDeviceAssignment() error = %v", err)
		}
		if got := mesh.NumDevices(); got != 4 {
			t.Errorf("NumDevices() = %d, want 4", got)
		}
		err = mesh.SetLogicalDeviceAssignment(4, 2, 1, 0)
		if err == nil {
			t.Error("SetLogicalDeviceAssignment() expected error for out of range device, got nil")
		}
	})

	t.Run("ComputeReplicaGroups", func(t *testing.T) {
		t.Run("2D mesh batch groups", func(t *testing.T) {
			mesh, err := shardy.NewDeviceMesh("mesh", []int{2, 2}, []string{"batch", "data"})
			if err != nil {
				t.Fatalf("NewDeviceMesh() error = %v", err)
			}

			// Example from comments: m.ComputeReplicaGroups([]string{"batch"}) -> [][]int{{0, 2}, {1, 3}}
			groups, err := mesh.ComputeReplicaGroups([]string{"batch"})
			if err != nil {
				t.Fatalf("ComputeReplicaGroups() error = %v", err)
			}
			expected := [][]int{{0, 2}, {1, 3}}
			if !reflect.DeepEqual(groups, expected) {
				t.Errorf("ComputeReplicaGroups() = %v, want %v", groups, expected)
			}
		})

		t.Run("2D mesh data groups", func(t *testing.T) {
			mesh, err := shardy.NewDeviceMesh("mesh", []int{2, 2}, []string{"batch", "data"})
			if err != nil {
				t.Fatalf("NewDeviceMesh() error = %v", err)
			}

			// Example from comments: m.ComputeReplicaGroups([]string{"data"}) -> [][]int{{0, 1}, {2, 3}}
			groups, err := mesh.ComputeReplicaGroups([]string{"data"})
			if err != nil {
				t.Fatalf("ComputeReplicaGroups() error = %v", err)
			}
			expected := [][]int{{0, 1}, {2, 3}}
			if !reflect.DeepEqual(groups, expected) {
				t.Errorf("ComputeReplicaGroups() = %v, want %v", groups, expected)
			}
		})

		t.Run("2D mesh global groups", func(t *testing.T) {
			mesh, err := shardy.NewDeviceMesh("mesh", []int{2, 2}, []string{"batch", "data"})
			if err != nil {
				t.Fatalf("NewDeviceMesh() error = %v", err)
			}

			// Example from comments: m.ComputeReplicaGroups([]string{"batch", "data"}) -> [][]int{{0, 1, 2, 3}}
			groups, err := mesh.ComputeReplicaGroups([]string{"batch", "data"})
			if err != nil {
				t.Fatalf("ComputeReplicaGroups() error = %v", err)
			}
			expected := [][]int{{0, 1, 2, 3}}
			if !reflect.DeepEqual(groups, expected) {
				t.Errorf("ComputeReplicaGroups() = %v, want %v", groups, expected)
			}
		})

		t.Run("1D mesh", func(t *testing.T) {
			mesh, err := shardy.NewDeviceMesh("mesh", []int{4}, []string{"replica"})
			if err != nil {
				t.Fatalf("NewDeviceMesh() error = %v", err)
			}

			groups, err := mesh.ComputeReplicaGroups([]string{"replica"})
			if err != nil {
				t.Fatalf("ComputeReplicaGroups() error = %v", err)
			}
			expected := [][]int{{0, 1, 2, 3}}
			if !reflect.DeepEqual(groups, expected) {
				t.Errorf("ComputeReplicaGroups() = %v, want %v", groups, expected)
			}
		})

		t.Run("3D mesh single axis", func(t *testing.T) {
			mesh, err := shardy.NewDeviceMesh("mesh", []int{2, 2, 2}, []string{"x", "y", "z"})
			if err != nil {
				t.Fatalf("NewDeviceMesh() error = %v", err)
			}

			// Groups along x axis: should split by y and z
			groups, err := mesh.ComputeReplicaGroups([]string{"x"})
			if err != nil {
				t.Fatalf("ComputeReplicaGroups() error = %v", err)
			}
			expected := [][]int{{0, 4}, {1, 5}, {2, 6}, {3, 7}}
			if !reflect.DeepEqual(groups, expected) {
				t.Errorf("ComputeReplicaGroups() = %v, want %v", groups, expected)
			}
		})

		t.Run("3D mesh two axes", func(t *testing.T) {
			mesh, err := shardy.NewDeviceMesh("mesh", []int{2, 2, 2}, []string{"x", "y", "z"})
			if err != nil {
				t.Fatalf("NewDeviceMesh() error = %v", err)
			}

			// Groups along x and y axes: should split by z
			groups, err := mesh.ComputeReplicaGroups([]string{"x", "y"})
			if err != nil {
				t.Fatalf("ComputeReplicaGroups() error = %v", err)
			}
			expected := [][]int{{0, 2, 4, 6}, {1, 3, 5, 7}}
			if !reflect.DeepEqual(groups, expected) {
				t.Errorf("ComputeReplicaGroups() = %v, want %v", groups, expected)
			}
		})

		t.Run("empty axes list", func(t *testing.T) {
			mesh, err := shardy.NewDeviceMesh("mesh", []int{2, 2}, []string{"batch", "data"})
			if err != nil {
				t.Fatalf("NewDeviceMesh() error = %v", err)
			}

			// Empty axes list: each device is its own group
			groups, err := mesh.ComputeReplicaGroups([]string{})
			if err != nil {
				t.Fatalf("ComputeReplicaGroups() error = %v", err)
			}
			expected := [][]int{{0}, {1}, {2}, {3}}
			if !reflect.DeepEqual(groups, expected) {
				t.Errorf("ComputeReplicaGroups() = %v, want %v", groups, expected)
			}
		})

		t.Run("non-existent axis", func(t *testing.T) {
			mesh, err := shardy.NewDeviceMesh("mesh", []int{2, 2}, []string{"batch", "data"})
			if err != nil {
				t.Fatalf("NewDeviceMesh() error = %v", err)
			}

			// A non-existent axis should return an error.
			_, err = mesh.ComputeReplicaGroups([]string{"nonexistent"})
			if err == nil {
				t.Error("ComputeReplicaGroups() expected error, got nil")
			}
		})
	})
}
