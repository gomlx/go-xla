package pjrt

import (
	"fmt"
	"os"
	"testing"

	"github.com/gomlx/go-xla/internal/protos/hlo"
	"google.golang.org/protobuf/proto"
)

type testFileInfo struct {
	name        string
	numOutputs  int
	testInputs  []float32
	wantOutputs [][]float32
}

var (
	testHLOPrograms = []testFileInfo{
		{
			name:        "test_hlo.pb",
			numOutputs:  1,
			testInputs:  []float32{1.0, 3.0},
			wantOutputs: [][]float32{{1.0}, {9.0}},
		},
		{
			name:        "test_tuple_hlo.pb",
			numOutputs:  2,
			testInputs:  []float32{1.0, 9.0},
			wantOutputs: [][]float32{{1.0, 1.0}, {81.0, 3.0}},
		},
	}
)

func TestPlugin_NewClient(t *testing.T) {
	plugin, err := GetPlugin(*FlagPluginName)
	requireNoError(t, err)
	fmt.Printf("Loaded %s\n", plugin)

	client, err := plugin.NewClient(nil)
	requireNoError(t, err, "Failed to create a client on %s", plugin)
	fmt.Printf("%s\n", client)

	devices := client.AddressableDevices()
	assertNotEmpty(t, devices, "No addressable devices for client on %s", plugin)

	err = client.Destroy()
	requireNoError(t, err, "Failed to destroy client on %s", plugin)
}

func TestCompileAndExecute(t *testing.T) {
	plugin, err := GetPlugin(*FlagPluginName)
	requireNoError(t, err)
	fmt.Printf("Loaded %s\n", plugin)

	client, err := plugin.NewClient(nil)
	requireNoError(t, err, "Failed to create a client on %s", plugin)
	fmt.Printf("%s\n", client)

	devices := client.AddressableDevices()
	assertNotEmpty(t, devices, "No addressable devices for client on %s", plugin)

	for _, programTest := range testHLOPrograms {
		fmt.Printf("Program: %s\n", programTest.name)

		// Load test program.
		hloBin, err := os.ReadFile(programTest.name)
		requireNoError(t, err)
		hloProto := &hlo.HloModuleProto{}
		requireNoError(t, proto.Unmarshal(hloBin, hloProto), "Unmarshalling HloModuleProto")
		//fmt.Printf("HloModuleProto: {\n%s}\n", prototext.Format(hloProto))

		// Compile program.
		loadedExec, err := client.Compile().WithHLO(hloBin).Done()
		requireNoError(t, err, "Failed to compile %q", programTest.name)
		fmt.Printf("\t> name=%s, #outputs=%d\n", loadedExec.Name, loadedExec.NumOutputs)

		for ii, input := range programTest.testInputs {
			buffer, err := client.BufferFromHost().FromRawData(ScalarToRaw(input)).Done()
			requireNoError(t, err, "Failed to transfer scalar %v", input)
			want := programTest.wantOutputs[ii]
			outputs, err := loadedExec.Execute(buffer).Done()
			requireNoError(t, err, "Failed to execute for %v", input)
			assertLen(t, outputs, len(want))
			if len(outputs) == 1 {
				got, err := BufferToScalar[float32](outputs[0])
				requireNoError(t, err, "Failed to transfer output to host for %v", input)
				fmt.Printf("\t> input=%f, output=%f, want=%f\n", input, got, want[0])
			}
			requireNoError(t, buffer.Destroy(), "Failed to destroy scalar buffer for %v", input)
		}

		// Destroy compiled executables.
		requireNoError(t, loadedExec.Destroy(), "Failed to destroy LoadedExecutable on %s", plugin)
	}

	// Destroy client.
	requireNoError(t, client.Destroy(), "Failed to destroy client on %s", plugin)
}
