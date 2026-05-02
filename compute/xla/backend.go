// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package xla

import (
	"fmt"
	"runtime"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/shapes"
	"github.com/gomlx/go-xla/pjrt"
	"github.com/pkg/errors"
	"k8s.io/klog/v2"
)

// Backend implements the XLA/PJRT compute.Backend for GoMLX.
type Backend struct {
	plugin             *pjrt.Plugin
	client             *pjrt.Client
	config, pluginName string
	hasSharedBuffers   bool
	capabilities       compute.Capabilities
	numDevices         int

	// DotGeneralUseTF32 controls whether to use TF32 for DotGeneral operations that are using float32.
	// (it can be faster in modern GPUs, and it's enabled by default)
	DotGeneralUseTF32 bool
}

// Compile-time check:
var (
	_ compute.DataInterface = (*Backend)(nil)
	_ compute.Backend       = (*Backend)(nil)
)

// CheckValid returns an error if the backend is not valid: if it's nil or has already been finalized.
func (backend *Backend) CheckValid() error {
	if backend == nil {
		return errors.Errorf("%q backend is nil", BackendName)
	}
	if backend.plugin == nil {
		return errors.Errorf("backend %q has already been finalized", BackendName)
	}
	return nil
}

// Name returns the short name of the backend. E.g.: "xla" for the StableHLO/PJRT plugin.
func (backend *Backend) Name() string {
	return BackendName
}

// PluginName returns the name of the PJRT plugin.
func (backend *Backend) PluginName() string {
	return backend.pluginName
}

// Config returns the config used to create the backend.
// If no configuration was used, it returns "".
func (backend *Backend) Config() string {
	return backend.config
}

// String returns Name().
func (backend *Backend) String() string {
	return backend.Name()
}

// Description is a longer description of the Backend that can be used to pretty-print.
func (backend *Backend) Description() string {
	if backend.CheckValid() != nil {
		return fmt.Sprintf("%s: in an invalid state!", BackendName)
	}
	return fmt.Sprintf("%s:%s - %s [%d device(s)]", BackendName, backend.pluginName, backend.plugin, backend.numDevices)
}

// NumDevices return the number of devices available for this Backend.
func (backend *Backend) NumDevices() int {
	if backend.CheckValid() != nil {
		return 0
	}
	return backend.numDevices
}

// DeviceDescription returns a description of the deviceNum.
func (backend *Backend) DeviceDescription(deviceNum compute.DeviceNum) string {
	if backend.CheckValid() != nil {
		return fmt.Sprintf("%s: in an invalid state!", BackendName)
	}
	if int(deviceNum) >= backend.numDevices || int(deviceNum) < 0 {
		return fmt.Sprintf("invalid deviceNum %d", deviceNum)
	}
	pjrtDevice := backend.client.AddressableDevices()[int(deviceNum)]
	pjrtDesc, err := pjrtDevice.GetDescription()
	if err != nil {
		return fmt.Sprintf("failed to get description for device %d: %v", deviceNum, err)
	}
	return fmt.Sprintf("%s [processId=%d]", pjrtDesc.DebugString(), pjrtDesc.ProcessIndex())
}

// Finalize releases all the associated resources immediately and makes the backend invalid.
func (backend *Backend) Finalize() {
	if backend.plugin == nil {
		return
	}
	if backend.client != nil {
		err := backend.client.Destroy()
		if err != nil {
			klog.Warningf("Failure while destroying PJRT client: %+v", err)
		}
		backend.client = nil
	}
	backend.plugin = nil
}

// IsFinalized returns true if the backend is in an invalid state.
func (backend *Backend) IsFinalized() bool {
	return backend == nil || backend.plugin == nil
}

// castToPJRT casts the buffer to pjrt.Buffer and panics if not possible.
func (backend *Backend) castToPJRT(buffer compute.Buffer) *pjrt.Buffer {
	b, ok := buffer.(*Buffer)
	if !ok {
		panic(errors.Errorf("buffer given is not a %q backend (pjrt) buffer", BackendName))
	}
	return b.pjrtBuffer
}

// Buffer implements compute.Buffer for XLA/PJRT.
type Buffer struct {
	backend    *Backend
	pjrtBuffer *pjrt.Buffer
}

// Compile-time check:
var _ compute.Buffer = (*Buffer)(nil)

// Backend implements compute.Buffer.
func (b *Buffer) Backend() compute.Backend {
	return b.backend
}

// Finalize implements compute.Buffer.
func (b *Buffer) Finalize() error {
	if err := b.backend.CheckValid(); err != nil {
		return errors.WithMessagef(err, "backend %q is invalid", BackendName)
	}
	err := b.pjrtBuffer.Destroy()
	if err != nil {
		return errors.WithMessagef(err, "backend %q: Buffer.Finalize", BackendName)
	}
	return nil
}

// Shape implements compute.Buffer.
func (b *Buffer) Shape() (shapes.Shape, error) {
	var noShape shapes.Shape
	if err := b.backend.CheckValid(); err != nil {
		return noShape, err
	}
	xlaDType, err := b.pjrtBuffer.DType()
	if err != nil {
		return noShape, errors.WithMessagef(err, "backend %q", BackendName)
	}
	dims, err := b.pjrtBuffer.Dimensions()
	if err != nil {
		return noShape, errors.WithMessagef(err, "backend %q", BackendName)
	}
	return shapes.Make(xlaDType, dims...), nil
}

// DeviceNum implements compute.Buffer.
func (b *Buffer) DeviceNum() (compute.DeviceNum, error) {
	if err := b.backend.CheckValid(); err != nil {
		return 0, err
	}
	device, err := b.pjrtBuffer.Device()
	if err != nil {
		return 0, errors.WithMessagef(err, "backend %q", BackendName)
	}
	num := b.pjrtBuffer.Client().NumForDevice(device)
	if num == -1 {
		return 0, errors.Errorf("backend %q: pjrt buffer stored on an unknown device!?", BackendName)
	}
	return compute.DeviceNum(num), nil
}

// ToFlatData implements compute.Buffer.
func (b *Buffer) ToFlatData(flat any) error {
	if err := b.backend.CheckValid(); err != nil {
		return err
	}
	shape, err := b.Shape()
	if err != nil {
		return err
	}
	if shape.IsZeroSize() {
		// No data to transfer.
		return nil
	}

	dstData := dtypes.UnsafeByteSliceFromAny(flat)
	var pinner runtime.Pinner
	pinner.Pin(b.pjrtBuffer)
	defer pinner.Unpin()
	err = b.pjrtBuffer.ToHost(dstData)
	if err != nil {
		return errors.WithMessagef(err, "backend %q: Buffer.ToFlatData", BackendName)
	}
	return nil
}

// Data implements compute.Buffer.
func (b *Buffer) Data() (flat any, err error) {
	if err := b.backend.CheckValid(); err != nil {
		return nil, err
	}
	if err = b.pjrtBuffer.Check(); err != nil {
		return nil, err
	}
	flat, err = b.pjrtBuffer.Data()
	if err != nil {
		return nil, errors.WithMessagef(err, "failed to access buffer data directly, maybe not supported by backend?")
	}
	return
}

// CopyToDevice implements compute.Buffer.
func (b *Buffer) CopyToDevice(deviceNum compute.DeviceNum) (compute.Buffer, error) {
	if err := b.backend.CheckValid(); err != nil {
		return nil, err
	}
	if err := b.pjrtBuffer.Check(); err != nil {
		return nil, err
	}
	srcDevice, err := b.pjrtBuffer.Device()
	if err != nil {
		return nil, errors.WithMessagef(err, "backend %q: Buffer.CopyToDevice failed to get device information", BackendName)
	}

	devices := b.backend.client.AddressableDevices()
	if deviceNum < 0 || int(deviceNum) >= len(devices) {
		return nil, errors.Errorf("deviceNum=%d not available for backend, only %d devices are available", deviceNum, len(devices))
	}
	device := devices[deviceNum]
	if srcDevice == device {
		return nil, errors.Errorf("backend %q: Buffer.CopyToDevice source and destination (#%d) "+
			"are the same device", BackendName, deviceNum)
	}

	var newPJRTBuf *pjrt.Buffer
	newPJRTBuf, err = b.pjrtBuffer.CopyToDevice(device)
	if err != nil {
		return nil, errors.WithMessagef(err, "backend %q: Buffer.CopyToDevice failed to copy "+
			"buffer to device", BackendName)
	}
	return &Buffer{backend: b.backend, pjrtBuffer: newPJRTBuf}, nil
}

// BufferFromFlatData transfers data from Go given as a flat slice (of the type corresponding to the shape DType)
// to the deviceNum, and returns the corresponding Buffer.
func (backend *Backend) BufferFromFlatData(deviceNum compute.DeviceNum, flat any, shape shapes.Shape) (compute.Buffer, error) {
	srcData := dtypes.UnsafeByteSliceFromAny(flat)
	pjrtBuffer, err := backend.client.BufferFromHost().
		FromRawData(srcData, shape.DType, shape.Dimensions).
		ToDeviceNum(int(deviceNum)).
		Done()
	if err != nil {
		return nil, errors.WithMessagef(err, "backend %q: BufferFromFlatData", BackendName)
	}
	return &Buffer{backend: backend, pjrtBuffer: pjrtBuffer}, nil
}

// HasSharedBuffers returns whether this PJRT plugin supports "shared buffers".
// In PJRT that means supporting pjrt.Client.CreateViewOfDeviceBuffer.
func (backend *Backend) HasSharedBuffers() bool {
	return backend.hasSharedBuffers
}

// NewSharedBuffer implements compute.Backend interface.
//
// For XLA this means allocating the aligned memory and calling pjrt.Client.CreateViewOfDeviceBuffer
// to create a buffer that shares the memory.
func (backend *Backend) NewSharedBuffer(deviceNum compute.DeviceNum, shape shapes.Shape) (buffer compute.Buffer, flat any, err error) {
	if err = backend.CheckValid(); err != nil {
		return
	}
	devices := backend.client.AddressableDevices()
	if deviceNum < 0 || int(deviceNum) >= len(devices) {
		err = errors.Errorf("deviceNum=%d not available for backend, only %d devices are available", deviceNum, len(devices))
		return
	}
	device := devices[deviceNum]
	var pjrtBuffer *pjrt.Buffer
	pjrtBuffer, flat, err = backend.client.NewSharedBuffer(shape.DType, shape.Dimensions, device)
	if err != nil {
		err = errors.WithMessagef(err, "backend %q NewSharedBuffer", BackendName)
		return
	}
	buffer = &Buffer{backend: backend, pjrtBuffer: pjrtBuffer}
	return
}

// Capabilities returns information about what is supported by this backend.
func (backend *Backend) Capabilities() compute.Capabilities {
	return backend.capabilities
}
