package pjrt

/*
#include "pjrt_c_api.h"
#include "gen_api_calls.h"
#include "gen_new_struct.h"
*/
import "C"
import (
	"reflect"
	"runtime"
	"slices"
	"sync"
	"sync/atomic"
	"unsafe"

	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/dtypes/gotype"
	"github.com/gomlx/go-xla/internal/shapeinference"
	"github.com/gomlx/go-xla/types/shapes"
	"github.com/pkg/errors"
	"k8s.io/klog/v2"
)

// Buffer is a reference to an on-device array storage (buffer).
type Buffer struct {
	client         *Client
	wrapper        *bufferWrapper
	wrapperCleanUp runtime.Cleanup

	// For "shared buffers", with a direct pointer to the underlying data.
	// This is nil for non-shared-buffers.
	isShared bool

	dimsSet bool // Whether dims is set.
	dims    []int

	dtypeSet bool // Whether dtype is set.
	dtype    dtypes.DType
	// DEBUG: creationStackTrace error
}

// bufferWrapper wraps the C/C++ data that requires clean up.
// It intentionally does NOT hold a pointer back to *Buffer so clientC can track it
// in an intrusive doubly-linked list without preventing Go GC of *Buffer.
type bufferWrapper struct {
	c      *C.PJRT_Buffer
	plugin *Plugin
	client *clientC

	// sharedMu guards sharedRawStorage so host shared memory is freed at most once
	// by Buffer.Destroy() or GC cleanup (and never eagerly unmapped by Client.Destroy()).
	sharedMu         sync.Mutex
	sharedRawStorage unsafe.Pointer

	// Intrusive doubly-linked list guarded by client.bufferShards[shard].mu.
	shard  uint8
	linked bool
	prev   *bufferWrapper
	next   *bufferWrapper
}

func (wrapper *bufferWrapper) IsValid() bool {
	return wrapper != nil && wrapper.c != nil
}

func (wrapper *bufferWrapper) freeSharedStorage() {
	if wrapper == nil {
		return
	}
	wrapper.sharedMu.Lock()
	raw := wrapper.sharedRawStorage
	wrapper.sharedRawStorage = nil
	wrapper.sharedMu.Unlock()
	if raw != nil {
		AlignedFree(raw)
	}
}

func destroyPJRTBufferC(plugin *Plugin, cBuf *C.PJRT_Buffer) error {
	if plugin == nil || plugin.api == nil || cBuf == nil {
		return nil
	}
	arena := plugin.getDefaultArena()
	defer plugin.returnArena(arena)
	args := arenaAlloc[C.PJRT_Buffer_Destroy_Args](arena)
	args.struct_size = C.PJRT_Buffer_Destroy_Args_STRUCT_SIZE
	args.buffer = cBuf
	err := toError(plugin, C.call_PJRT_Buffer_Destroy(plugin.api, args))
	buffersAlive.Add(-1)
	return err
}

// destroyLocked destroys the PJRT_Buffer and frees any shared storage.
// Caller MUST already hold wrapper.client.lifecycle.Enter().
func (wrapper *bufferWrapper) destroyLocked() error {
	if wrapper == nil {
		return nil
	}
	defer wrapper.freeSharedStorage()
	if wrapper.client == nil {
		return nil
	}
	cBuf, plugin := wrapper.client.untrackAndDetachBuffer(wrapper)
	if cBuf == nil || plugin == nil || plugin.api == nil {
		return nil
	}
	defer runtime.KeepAlive(wrapper)
	return destroyPJRTBufferC(plugin, cBuf)
}

func (wrapper *bufferWrapper) Destroy() error {
	if wrapper == nil {
		return nil
	}
	defer wrapper.freeSharedStorage()
	client := wrapper.client
	if client == nil || !client.lifecycle.Enter() {
		// Client has already been destroyed (which already destroyed all tracked PJRT_Buffers).
		return nil
	}
	defer client.lifecycle.Leave()
	cBuf, plugin := client.untrackAndDetachBuffer(wrapper)
	if cBuf == nil || plugin == nil || plugin.api == nil {
		// Already destroyed, no-op.
		return nil
	}
	defer runtime.KeepAlive(wrapper)
	return destroyPJRTBufferC(plugin, cBuf)
}

var buffersAlive atomic.Int64

// BuffersAlive returns the number of PJRT Buffers in memory and currently tracked by gopjrt.
func BuffersAlive() int64 {
	return buffersAlive.Load()
}

// newBufferLocked creates a Buffer, tracks its detached wrapper on client.client,
// and registers it for GC cleanup. Caller MUST already hold client.client.lifecycle.Enter().
func newBufferLocked(client *Client, cBuffer *C.PJRT_Buffer) *Buffer {
	w := &bufferWrapper{
		plugin: client.plugin,
		c:      cBuffer,
		client: client.client,
	}
	client.client.trackBuffer(w)
	b := &Buffer{
		client:  client,
		wrapper: w,
		// DEBUG: creationStackTrace: errors.New("bufferCreation"),
	}
	buffersAlive.Add(1)
	b.wrapperCleanUp = runtime.AddCleanup(b, func(wrapper *bufferWrapper) {
		err := wrapper.Destroy()
		if err != nil {
			klog.Errorf("pjrt.Buffer.Destroy failed: %v", err)
		}
	}, b.wrapper)
	return b
}

// destroyLocked destroys b when caller already holds b.wrapper.client.lifecycle.Enter()
// (used for donated inputs inside LoadedExecutable.Execute().Done()).
func (b *Buffer) destroyLocked() error {
	if b == nil || b.wrapper == nil {
		return nil
	}
	err := b.wrapper.destroyLocked()
	b.wrapperCleanUp.Stop()
	return err
}

// Destroy the Buffer, release resources, and Buffer is no longer valid.
// This is automatically called if Buffer is garbage collected.
// Safe to call multiple times or after the parent Client has been destroyed.
func (b *Buffer) Destroy() error {
	if b == nil || b.wrapper == nil {
		return nil
	}
	err := b.wrapper.Destroy()
	b.wrapperCleanUp.Stop()
	return err
}

// Check returns an error if the Buffer is invalid or its Client has been destroyed.
func (b *Buffer) Check() error {
	if b == nil || b.wrapper == nil || b.wrapper.client == nil {
		return errors.New("Buffer is invalid: it is nil or uninitialized")
	}
	if b.wrapper.client.lifecycle.IsFinalized() {
		return errClientDestroyed("Buffer.Check")
	}
	if !b.wrapper.IsValid() {
		return errors.New("Buffer is invalid: it has been destroyed")
	}
	return nil
}

// enterClient acquires the parent client's lifecycle read lock and verifies that b is still valid.
// When err == nil, caller MUST call b.wrapper.client.lifecycle.Leave().
func (b *Buffer) enterClient(op string) (*Plugin, error) {
	if b == nil || b.wrapper == nil || b.wrapper.client == nil {
		return nil, errors.Errorf("Buffer.%s: buffer is nil or uninitialized", op)
	}
	if !b.wrapper.client.lifecycle.Enter() {
		return nil, errClientDestroyed("Buffer." + op)
	}
	if !b.wrapper.IsValid() || b.wrapper.plugin == nil {
		b.wrapper.client.lifecycle.Leave()
		return nil, errors.Errorf("Buffer.%s: buffer has been destroyed", op)
	}
	return b.wrapper.plugin, nil
}

// Dimensions of the Buffer.
// The buffer owns the returned slice to avoid creating a copy. Don't change it.
func (b *Buffer) Dimensions() (dims []int, err error) {
	plugin, err := b.enterClient("Dimensions")
	if err != nil {
		return nil, err
	}
	defer b.wrapper.client.lifecycle.Leave()
	return b.dimensionsLocked(plugin)
}

func (b *Buffer) dimensionsLocked(plugin *Plugin) (dims []int, err error) {
	if b.dimsSet {
		return b.dims, nil
	}
	defer runtime.KeepAlive(b)

	arena := plugin.getDefaultArena()
	defer plugin.returnArena(arena)
	args := arenaAlloc[C.PJRT_Buffer_Dimensions_Args](arena)
	args.struct_size = C.PJRT_Buffer_Dimensions_Args_STRUCT_SIZE
	args.buffer = b.wrapper.c
	err = toError(plugin, C.call_PJRT_Buffer_Dimensions(plugin.api, args))
	if err != nil {
		return
	}
	if args.num_dims == 0 {
		return // dims = nil
	}
	b.dims = slices.Clone(cDataToSlice[int](unsafe.Pointer(args.dims), int(args.num_dims)))
	b.dimsSet = true
	return b.dims, nil
}

// DType of the Buffer (PJRT_Buffer_ElementType).
func (b *Buffer) DType() (dtype dtypes.DType, err error) {
	dtype = dtypes.InvalidDType
	plugin, err := b.enterClient("DType")
	if err != nil {
		return dtype, err
	}
	defer b.wrapper.client.lifecycle.Leave()
	return b.dtypeLocked(plugin)
}

func (b *Buffer) dtypeLocked(plugin *Plugin) (dtype dtypes.DType, err error) {
	dtype = dtypes.InvalidDType
	defer runtime.KeepAlive(b)
	if b.dtypeSet {
		return b.dtype, nil
	}

	arena := plugin.getDefaultArena()
	defer plugin.returnArena(arena)
	args := arenaAlloc[C.PJRT_Buffer_ElementType_Args](arena)
	args.struct_size = C.PJRT_Buffer_ElementType_Args_STRUCT_SIZE
	args.buffer = b.wrapper.c
	err = toError(plugin, C.call_PJRT_Buffer_ElementType(plugin.api, args))
	if err != nil {
		return
	}
	dtype = dtypes.DType(args._type)
	b.dtype = dtype
	b.dtypeSet = true
	return
}

// Device returns the device the buffer is stored.
func (b *Buffer) Device() (device *Device, err error) {
	plugin, err := b.enterClient("Device")
	if err != nil {
		return nil, err
	}
	defer b.wrapper.client.lifecycle.Leave()
	defer runtime.KeepAlive(b)

	arena := plugin.getDefaultArena()
	defer plugin.returnArena(arena)
	args := arenaAlloc[C.PJRT_Buffer_Device_Args](arena)
	args.struct_size = C.PJRT_Buffer_Device_Args_STRUCT_SIZE
	args.buffer = b.wrapper.c
	err = toError(plugin, C.call_PJRT_Buffer_Device(plugin.api, args))
	if err != nil {
		return
	}
	device = newDevice(b.client, args.device)
	return
}

// Client returns the client that created this Buffer.
func (b *Buffer) Client() *Client {
	if b == nil {
		return nil
	}
	return b.client
}

// ScalarToRaw generates the raw values needed by BufferFromHostConfig.FromRawData to feed a simple scalar value.
func ScalarToRaw[T gotype.Supported](value T) ([]byte, dtypes.DType, []int) {
	dtype := dtypes.FromGenericsType[T]()
	rawSlice := unsafe.Slice((*byte)(unsafe.Pointer(&value)), int(unsafe.Sizeof(value)))
	return rawSlice, dtype, nil // empty dimensions for scalar
}

// Size returns the size in bytes if required for the buffer to be transferred with ToHost.
func (b *Buffer) Size() (int, error) {
	plugin, err := b.enterClient("Size")
	if err != nil {
		return 0, err
	}
	defer b.wrapper.client.lifecycle.Leave()
	defer runtime.KeepAlive(b)

	arena := plugin.getDefaultArena()
	defer plugin.returnArena(arena)

	// It uses a PJRT_Buffer_ToHostBuffer_Args but it doesn't transfer, only inquire about size.
	args := arenaAlloc[C.PJRT_Buffer_ToHostBuffer_Args](arena)
	args.struct_size = C.PJRT_Buffer_ToHostBuffer_Args_STRUCT_SIZE
	args.src = b.wrapper.c
	args.dst = nil // Don't transfer, only inquire about size.
	err = toError(plugin, C.call_PJRT_Buffer_ToHostBuffer(plugin.api, args))
	if err != nil {
		return 0, errors.WithMessage(err, "Failed to call PJRT_Buffer_ToHostBuffer for inquiring size of the buffer")
	}
	return int(args.dst_size), nil
}

// BufferToScalar is a generic function that transfer a Buffer back to host as a scalar of the given type.
func BufferToScalar[T gotype.Supported](b *Buffer) (value T, err error) {
	dst := unsafe.Slice((*byte)(unsafe.Pointer(&value)), unsafe.Sizeof(value))
	err = b.ToHost(dst)
	return
}

// ScalarToBuffer transfers the scalar value to a Buffer on the default device.
//
// It is a shortcut to Client.BufferFromHost call with default parameters.
// If you need more control where the value will be used you'll have to use Client.BufferFromHost instead.
func ScalarToBuffer[T gotype.Supported](client *Client, value T) (b *Buffer, err error) {
	dtype := dtypes.FromGenericsType[T]()
	src := unsafe.Slice((*byte)(unsafe.Pointer(&value)), unsafe.Sizeof(value))
	return client.BufferFromHost().FromRawData(src, dtype, nil).Done()
}

// ScalarToBufferOnDeviceNum transfers the scalar value to a Buffer on the given device.
//
// It is a shortcut to Client.BufferFromHost call with default parameters.
// If you need more control where the value will be used you'll have to use Client.BufferFromHost instead.
func ScalarToBufferOnDeviceNum[T gotype.Supported](client *Client, deviceNum int, value T) (b *Buffer, err error) {
	dtype := dtypes.FromGenericsType[T]()
	src := unsafe.Slice((*byte)(unsafe.Pointer(&value)), unsafe.Sizeof(value))
	return client.BufferFromHost().FromRawData(src, dtype, nil).ToDeviceNum(deviceNum).Done()
}

// ArrayToBuffer transfer a slice to a Buffer on the default device.
// The underlying array is provided with its flat values as a slice, and the underlying dimensions.
//
// It is a shortcut to Client.BufferFromHost call with default parameters.
// If you need more control where the value will be used you'll have to use Client.BufferFromHost instead.
func ArrayToBuffer[T gotype.Supported](client *Client, flatValues []T, dimensions ...int) (b *Buffer, err error) {
	if len(dimensions) == 0 && len(flatValues) != 1 {
		return nil, errors.Errorf("ArrayToBuffer not given any dimensions (indicating a scalar), but len(flatValues) == %d", len(flatValues))
	}
	return client.BufferFromHost().FromFlatDataWithDimensions(flatValues, dimensions).Done()
}

// BufferToArray transfers the buffer to an array defined by a slice with its flat values, and returns also its underlying dimensions.
func BufferToArray[T gotype.Supported](buffer *Buffer) (flatValues []T, dimensions []int, err error) {
	if err = buffer.Check(); err != nil {
		return
	}
	var dtype dtypes.DType
	dtype, err = buffer.DType()
	if err != nil {
		return
	}
	requestedDType := dtypes.FromGenericsType[T]()
	if dtype != requestedDType {
		var dummy T
		err = errors.Errorf("called BufferToArray[%T](...), but underlying buffer has dtype %s", dummy, dtype)
		return
	}
	dimensions, err = buffer.Dimensions()
	if err != nil {
		return
	}
	totalSize := 1
	for _, dim := range dimensions {
		totalSize *= dim
	}
	if totalSize <= 0 {
		// Odd empty buffer (likely one of the dimensions was 0), we return nil for the flatValues, the reported dimensions
		// and no error.
		return
	}
	flatValues = make([]T, totalSize)
	flatValuesPtr := unsafe.SliceData(flatValues)
	dst := unsafe.Slice((*byte)(
		unsafe.Pointer(flatValuesPtr)),
		totalSize*int(unsafe.Sizeof(flatValues[0])))
	err = buffer.ToHost(dst)
	return
}

// ToFlatDataAndDimensions transfers the buffer to a flat slice and returns also its underlying dimensions.
//
// Similar to the generic BufferToArray[T], but this returns an anonymous typed (`any`) flat slice instead of using generics.
func (b *Buffer) ToFlatDataAndDimensions() (flat any, dimensions []int, err error) {
	if err = b.Check(); err != nil {
		return
	}
	var dtype dtypes.DType
	dtype, err = b.DType()
	if err != nil {
		return
	}

	dimensions, err = b.Dimensions()
	if err != nil {
		return
	}
	totalSize := 1
	for _, dim := range dimensions {
		totalSize *= dim
	}
	if totalSize <= 0 {
		// Odd empty b (likely one of the dimensions was 0), we return nil for the flat, the reported dimensions
		// and no error.
		return
	}
	goType := dtype.GoType()
	flatV := reflect.MakeSlice(reflect.SliceOf(goType), totalSize, totalSize)
	element0 := flatV.Index(0)
	flatValuesPtr := element0.Addr().UnsafePointer()
	sizeBytes := uintptr(flatV.Len()) * element0.Type().Size()

	dst := unsafe.Slice((*byte)(flatValuesPtr), sizeBytes)
	err = b.ToHost(dst)
	flat = flatV.Interface()
	return
}

// CopyToDevice copies the buffer to the given device and returns a new buffer.
// The original buffer is not affected.
func (b *Buffer) CopyToDevice(dstDevice *Device) (*Buffer, error) {
	plugin, err := b.enterClient("CopyToDevice")
	if err != nil {
		return nil, err
	}
	defer b.wrapper.client.lifecycle.Leave()
	if dstDevice == nil || dstDevice.cDevice == nil {
		return nil, errors.New("destination device is nil")
	}
	defer runtime.KeepAlive(b)
	defer runtime.KeepAlive(dstDevice)

	arena := plugin.getDefaultArena()
	defer plugin.returnArena(arena)

	args := arenaAlloc[C.PJRT_Buffer_CopyToDevice_Args](arena)
	args.struct_size = C.PJRT_Buffer_CopyToDevice_Args_STRUCT_SIZE
	args.buffer = b.wrapper.c
	args.dst_device = dstDevice.cDevice

	err = toError(plugin, C.call_PJRT_Buffer_CopyToDevice(plugin.api, args))
	if err != nil {
		return nil, errors.WithMessagef(err, "failed to copy buffer to device %v", dstDevice)
	}

	newBuff := newBufferLocked(b.client, args.dst_buffer)
	return newBuff, nil
}

// Bitcast bitcasts the buffer to a new type, following the same expansion/contraction rules
// as used by shapeinference.BitcastConvert to figure out the output shape.
func (b *Buffer) Bitcast(dtype dtypes.DType) (*Buffer, error) {
	plugin, err := b.enterClient("Bitcast")
	if err != nil {
		return nil, err
	}
	defer b.wrapper.client.lifecycle.Leave()

	currentDType, err := b.dtypeLocked(plugin)
	if err != nil {
		return nil, err
	}
	currentDims, err := b.dimensionsLocked(plugin)
	if err != nil {
		return nil, err
	}

	currentShape := shapes.Make(currentDType, currentDims...)
	targetShape, err := shapeinference.BitcastConvert(currentShape, dtype)
	if err != nil {
		return nil, errors.WithMessagef(err, "failed to determine target shape for bitcast from %s to %s", currentShape, dtype)
	}

	defer runtime.KeepAlive(b)
	arena := plugin.getDefaultArena()
	defer plugin.returnArena(arena)

	args := arenaAlloc[C.PJRT_Buffer_Bitcast_Args](arena)
	args.struct_size = C.PJRT_Buffer_Bitcast_Args_STRUCT_SIZE
	args.buffer = b.wrapper.c
	args.element_type = C.PJRT_Buffer_Type(dtype)

	if len(targetShape.Dimensions) > 0 {
		cDims := arenaAllocSlice[C.int64_t](arena, len(targetShape.Dimensions))
		for i, dim := range targetShape.Dimensions {
			cDims[i] = C.int64_t(dim)
		}
		args.dims = (*C.int64_t)(unsafe.Pointer(&cDims[0]))
		args.num_dims = C.size_t(len(targetShape.Dimensions))
	}
	args.device_layout = nil // use default layout

	if plugin.api.PJRT_Buffer_Bitcast == nil {
		return nil, errors.Errorf("PJRT_Buffer_Bitcast is not supported by the current plugin version %v", plugin)
	}

	err = toError(plugin, C.call_PJRT_Buffer_Bitcast(plugin.api, args))
	if err != nil {
		return nil, errors.WithMessage(err, "failed to call PJRT_Buffer_Bitcast")
	}

	return newBufferLocked(b.client, args.out_buffer), nil
}
