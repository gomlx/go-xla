package pjrt

import (
	"fmt"
	"testing"
)

func TestClient_Devices(t *testing.T) {
	plugin, err := GetPlugin(*FlagPluginName)
	requireNoError(t, err)
	fmt.Printf("Loaded %s\n", plugin)
	client, err := plugin.NewClient(nil)
	requireNoError(t, err, "Failed to create a client on %s", plugin)
	fmt.Printf("\t%s\n", client)

	devices, err := client.AllDevices()
	requireNoError(t, err, "Failed to list devices for %s", client)

	addressableDevices := client.AddressableDevices()
	fmt.Printf("\t%d devices, %d addressable\n", len(devices), len(addressableDevices))

	if client.ProcessIndex() == 0 {
		if len(devices) != len(addressableDevices) {
			t.Fatalf("In single-process client (process index==0), all devices should be addressable, but only %d out of %d are",
				len(addressableDevices), len(devices))
		}
	}

	var countAddressable int
	for _, d := range devices {
		isAddr, err := d.IsAddressable()
		requireNoError(t, err)
		if isAddr {
			countAddressable++
		}
		desc, err := d.GetDescription()
		requireNoError(t, err)
		fmt.Printf("\t\tDevice Local Hardware Id %d: %s\n", d.LocalHardwareID(), desc.DebugString())
	}
	assertEqual(t, countAddressable, len(addressableDevices))
	requireNoError(t, client.Destroy())
}
