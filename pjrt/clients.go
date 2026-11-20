package pjrt

/*
#include "pjrt_c_api.h"
#include "gen_api_calls.h"
#include "gen_new_struct.h"
*/
import "C"
import (
	"fmt"
	"runtime"
	"sync"
	"sync/atomic"
	"unsafe"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/support/xsync"
	"github.com/pkg/errors"
	"k8s.io/klog/v2"
)

func pjrtClientPlatformName(plugin *Plugin, client *Client) (string, error) {
	args := C.new_PJRT_Client_PlatformName_Args()
	defer cFree(args)
	args.client = client.client.c
	err := toError(plugin, C.call_PJRT_Client_PlatformName(plugin.api, args))
	if err != nil {
		return "", err
	}
	return cCharArray(args.platform_name, args.platform_name_size), nil
}

func pjrtClientPlatformVersion(plugin *Plugin, client *Client) (string, error) {
	args := C.new_PJRT_Client_PlatformVersion_Args()
	defer cFree(args)
	args.client = client.client.c
	err := toError(plugin, C.call_PJRT_Client_PlatformVersion(plugin.api, args))
	if err != nil {
		return "", err
	}
	return cCharArray(args.platform_version, args.platform_version_size), nil
}

func pjrtClientProcessIndex(plugin *Plugin, client *Client) (int, error) {
	args := C.new_PJRT_Client_ProcessIndex_Args()
	defer cFree(args)
	args.client = client.client.c
	err := toError(plugin, C.call_PJRT_Client_ProcessIndex(plugin.api, args))
	if err != nil {
		return -1, err
	}
	return int(args.process_index), nil
}

func pjrtClientDevices(plugin *Plugin, client *Client) ([]*Device, error) {
	args := C.new_PJRT_Client_Devices_Args()
	defer cFree(args)
	args.client = client.client.c
	err := toError(plugin, C.call_PJRT_Client_Devices(plugin.api, args))
	if err != nil {
		return nil, err
	}
	cDevices := cDataToSlice[*C.PJRT_Device](unsafe.Pointer(args.devices), int(args.num_devices))
	devices := make([]*Device, len(cDevices))
	for ii, d := range cDevices {
		devices[ii] = newDevice(client, d)
	}
	return devices, nil
}

func pjrtClientAddressableDevices(plugin *Plugin, client *Client) ([]*Device, error) {
	args := C.new_PJRT_Client_AddressableDevices_Args()
	defer cFree(args)
	args.client = client.client.c
	err := toError(plugin, C.call_PJRT_Client_AddressableDevices(plugin.api, args))
	if err != nil {
		return nil, err
	}
	cDevices := cDataToSlice[*C.PJRT_Device](unsafe.Pointer(args.addressable_devices), int(args.num_addressable_devices))
	devices := make([]*Device, len(cDevices))
	for ii, d := range cDevices {
		devices[ii] = newDevice(client, d)
	}
	return devices, nil
}

func pjrtClientDefaultDeviceAssignment(plugin *Plugin, client *Client, numReplicas, numPartitions int) ([]int, error) {
	args := C.new_PJRT_Client_DefaultDeviceAssignment_Args()
	defer cFree(args)
	args.client = client.client.c
	args.num_replicas = C.int(numReplicas)
	args.num_partitions = C.int(numPartitions)
	assignmentSize := numReplicas * numPartitions
	args.default_assignment_size = (C.size_t)(assignmentSize)
	args.default_assignment = cMallocArray[C.int](assignmentSize)
	defer cFree(args.default_assignment)
	err := toError(plugin, C.call_PJRT_Client_DefaultDeviceAssignment(plugin.api, args))
	if err != nil {
		return nil, err
	}
	cAssignment := cDataToSlice[C.int](unsafe.Pointer(args.default_assignment), assignmentSize)
	assignment := make([]int, assignmentSize)
	for i, v := range cAssignment {
		assignment[i] = int(v)
	}
	return assignment, nil
}

// pjrtClientCompile compiles the program. Make sure that both the program and the compileOptionsProto
// are pinned until the C function returns.
func pjrtClientCompile(plugin *Plugin, client *Client, program []byte, programFormat string,
	compileOptionsProto []byte) (*LoadedExecutable, error) {

	// Create the program struct.
	var cProgram *C.PJRT_Program
	cProgram = C.new_PJRT_Program()
	defer cFree(cProgram)
	cProgramFormat := C.CString(programFormat)
	defer cFree(cProgramFormat)
	cProgram.format = cProgramFormat
	cProgram.format_size = (C.size_t)(len(programFormat))
	cProgram.code = (*C.char)(unsafe.Pointer(unsafe.SliceData(program)))
	cProgram.code_size = (C.size_t)(len(program))

	// Create args for call.
	args := C.new_PJRT_Client_Compile_Args()
	defer cFree(args)
	args.client = client.client.c
	args.program = cProgram

	if len(compileOptionsProto) != 0 {
		args.compile_options = (*C.char)(C.CBytes(compileOptionsProto))
		//args.compile_options = (*C.char)(unsafe.Pointer(unsafe.SliceData(compileOptionsProto)))
		args.compile_options_size = (C.size_t)(len(compileOptionsProto))
		defer cFree(args.compile_options)
	}
	cErr := C.call_PJRT_Client_Compile(plugin.api, args)
	runtime.KeepAlive(program) // Makes sure it is alive during the C call.
	err := toError(plugin, cErr)
	if err != nil {
		return nil, err
	}
	return newLoadedExecutableLocked(plugin, client, args.executable)
}

const numBufferShards = 32

type bufferShard struct {
	mu   sync.Mutex
	head *bufferWrapper
}

// Client manages the resources of one device: its buffers, compilation and execution of HLO code.
type Client struct {
	plugin                    *Plugin
	client                    *clientC
	cleanup                   runtime.Cleanup
	platform, platformVersion string
	processIndex              int
	addressableDevices        []*Device
	allowBufferViews          bool
}

type clientC struct {
	// c holds the pointer to the C/C++ structure.
	c      *C.PJRT_Client
	plugin *Plugin

	// lifecycle coordinates in-flight PJRT calls against Client.Destroy().
	lifecycle xsync.Lifecycle

	// bufferShards holds a 32-shard intrusive doubly-linked list of live *bufferWrapper
	// instances so Client.Destroy() can destroy all PJRT_Buffer handles before
	// PJRT_Client_Destroy without preventing Go GC of *Buffer.
	bufferShards [numBufferShards]bufferShard
	nextShard    atomic.Uint32

	// execMu guards execs, tracking live detached *loadedExecutableC wrappers so
	// Client.Destroy() destroys all executables before PJRT_Client_Destroy without
	// preventing Go GC of *LoadedExecutable.
	execMu sync.Mutex
	execs  map[*loadedExecutableC]struct{}
}

func errClientDestroyed(op string) error {
	return errors.Wrapf(compute.ErrBackendFinalized, "pjrt.Client.%s: client has been destroyed", op)
}

func (cc *clientC) trackBuffer(w *bufferWrapper) {
	shardIdx := uint8(cc.nextShard.Add(1) % numBufferShards)
	w.shard = shardIdx
	s := &cc.bufferShards[shardIdx]
	s.mu.Lock()
	w.prev = nil
	w.next = s.head
	if s.head != nil {
		s.head.prev = w
	}
	s.head = w
	w.linked = true
	s.mu.Unlock()
}

// untrackAndDetachBuffer unlinks w from its shard and atomically detaches its C PJRT_Buffer
// pointer so at most one caller destroys it.
func (cc *clientC) untrackAndDetachBuffer(w *bufferWrapper) (*C.PJRT_Buffer, *Plugin) {
	s := &cc.bufferShards[w.shard]
	s.mu.Lock()
	if w.linked {
		if w.prev != nil {
			w.prev.next = w.next
		} else if s.head == w {
			s.head = w.next
		}
		if w.next != nil {
			w.next.prev = w.prev
		}
		w.prev = nil
		w.next = nil
		w.linked = false
	}
	cBuf := w.c
	plugin := w.plugin
	w.c = nil
	w.plugin = nil
	s.mu.Unlock()
	return cBuf, plugin
}

func (cc *clientC) drainAndDestroyBuffersLocked() {
	for i := range cc.bufferShards {
		s := &cc.bufferShards[i]
		s.mu.Lock()
		head := s.head
		s.head = nil
		for w := head; w != nil; {
			next := w.next
			w.prev = nil
			w.next = nil
			w.linked = false
			cBuf := w.c
			plugin := w.plugin
			w.c = nil
			w.plugin = nil
			w = next
			if cBuf != nil && plugin != nil && plugin.api != nil {
				// Release shard lock? No other goroutine holds lifecycle.Enter() during Finalize,
				// so calling destroyPJRTBufferC directly while walking the detached chain is safe.
				if err := destroyPJRTBufferC(plugin, cBuf); err != nil {
					klog.Warningf("Failed to destroy pjrt.Buffer during Client.Destroy: %+v", err)
				}
			}
		}
		s.mu.Unlock()
	}
}

func (cc *clientC) trackExecutable(w *loadedExecutableC) {
	cc.execMu.Lock()
	if cc.execs == nil {
		cc.execs = make(map[*loadedExecutableC]struct{})
	}
	cc.execs[w] = struct{}{}
	cc.execMu.Unlock()
}

func (cc *clientC) untrackExecutable(w *loadedExecutableC) {
	cc.execMu.Lock()
	delete(cc.execs, w)
	cc.execMu.Unlock()
}

func (cc *clientC) drainAndDestroyExecutablesLocked() {
	cc.execMu.Lock()
	execs := cc.execs
	cc.execs = nil
	cc.execMu.Unlock()
	for w := range execs {
		if err := w.destroyLocked(); err != nil {
			klog.Warningf("Failed to destroy pjrt.LoadedExecutable during Client.Destroy: %+v", err)
		}
	}
}

// newClient is called by Plugin.NewClient to create a new PJRT_Client wrapper.
func newClient(plugin *Plugin, options NamedValuesMap) (*Client, error) {
	// Create C.PJRT_Client object.
	args := C.new_PJRT_Client_Create_Args()
	defer cFree(args)
	var err error
	args.create_options, args.num_options, err = options.mallocArrayPJRT_NamedValue()
	if err != nil {
		return nil, errors.WithMessagef(err, "invalid options when creating a new pjrt.Client")
	}
	// No callback support yet, so we leave the various PJRT_KeyValue... fields empty.
	err = toError(plugin, C.call_PJRT_Client_Create(plugin.api, args))
	if err != nil {
		return nil, err
	}

	// Prepare the Client object: not all initializations are fatal to the construction of the client.
	c := &Client{
		plugin: plugin,
		client: &clientC{
			c:      args.client,
			plugin: plugin,
		},
	}
	c.platform, err = pjrtClientPlatformName(plugin, c)
	if err != nil {
		// Non-fatal
		klog.Errorf("Failed to retrieve client platform name (plugin %s): %v", plugin, err)
	}
	c.platformVersion, err = pjrtClientPlatformVersion(plugin, c)
	if err != nil {
		// Non-fatal
		klog.Errorf("Failed to retrieve client platform version (plugin %s): %v", plugin, err)
	}
	c.processIndex, err = pjrtClientProcessIndex(plugin, c)
	if err != nil {
		// Non-fatal
		klog.Errorf("Failed to retrieve client process index (plugin %s): %v", plugin, err)
	}
	c.addressableDevices, err = pjrtClientAddressableDevices(plugin, c)
	if err != nil {
		// Fatal
		err = errors.WithMessagef(err, "failed to retrieve addressable devices for new client (plugin %s) -- can't use client with no addressable device", plugin)
		_ = c.client.Destroy()
		return nil, err
	}

	// Register cleanup.
	c.cleanup = runtime.AddCleanup(c, func(client *clientC) {
		err := client.Destroy()
		if err != nil {
			klog.Errorf("Failed to destroy client (plugin %s): %v", plugin, err)
		}
	}, c.client)
	return c, nil
}

// Plugin returns the Plugin from which the Client was created.
func (c *Client) Plugin() *Plugin {
	return c.plugin
}

func (client *clientC) Destroy() error {
	if client == nil {
		return nil
	}
	var destroyErr error
	client.lifecycle.Finalize(func() {
		defer runtime.KeepAlive(client)
		// Destroy all tracked executables and buffers BEFORE destroying PJRT_Client,
		// as required by the PJRT C API contract.
		client.drainAndDestroyExecutablesLocked()
		client.drainAndDestroyBuffersLocked()

		plugin := client.plugin
		if plugin == nil || plugin.api == nil || client.c == nil {
			client.c = nil
			return
		}
		args := C.new_PJRT_Client_Destroy_Args()
		defer cFree(args)
		args.client = client.c
		destroyErr = toError(plugin, C.call_PJRT_Client_Destroy(plugin.api, args))
		client.c = nil
	})
	return destroyErr
}

// IsValid returns if client has been properly created and not yet destroyed.
func (c *Client) IsValid() bool {
	return c != nil && c.client != nil && !c.client.lifecycle.IsFinalized()
}

// Destroy the client, release resources, and Client is no longer valid.
// It waits for any in-flight PJRT calls on this client to finish, destroys all tracked
// executables and buffers, and then destroys the underlying PJRT_Client.
// This is automatically called if Client is garbage collected.
func (c *Client) Destroy() error {
	if c == nil || c.client == nil {
		// Already destroyed, no-op.
		return nil
	}
	defer runtime.KeepAlive(c)
	c.cleanup.Stop()
	return c.client.Destroy()
}

// String implements fmt.Stringer.
func (c *Client) String() string {
	if c.client == nil {
		return "Invalid client"
	}

	pid := c.ProcessIndex()
	var pidStr string
	if pid == 0 {
		pidStr = "single-process"
	} else {
		pidStr = fmt.Sprintf("pid=%d", pid)
	}
	return fmt.Sprintf("Client[plugin=%q, platform=%q, %s, %d device(s)]",
		c.plugin.Name(), c.Platform()+" - "+c.PlatformVersion(), pidStr, len(c.addressableDevices))
}

// Platform returns the name of the client platform.
func (c *Client) Platform() string {
	return c.platform
}

// PlatformVersion returns the version of the client platform.
func (c *Client) PlatformVersion() string {
	return c.platformVersion
}

// ProcessIndex returns the process index of the client platform.
// Always 0 in single-process settings.
func (c *Client) ProcessIndex() int {
	return c.processIndex
}

// AllDevices returns a list of all devices visible to the runtime, including addressable
// and non-addressable devices.
//
// Usually, you want to use the AddressableDevices method instead.
func (c *Client) AllDevices() ([]*Device, error) {
	if c == nil || c.client == nil || !c.client.lifecycle.Enter() {
		return nil, errClientDestroyed("AllDevices")
	}
	defer c.client.lifecycle.Leave()
	return pjrtClientDevices(c.plugin, c)
}

// AddressableDevices returns a list of devices addressable to the client.
// Addressable devices are those that the client can issue commands to.
// All devices are addressable in a single-process environment (Client.ProcessIndex() == 0).
//
// The Client owns the returned slice and the devices. Don't change them.
func (c *Client) AddressableDevices() []*Device {
	return c.addressableDevices
}

// NumDevices returns the number of addressable devices.
func (c *Client) NumDevices() int {
	return len(c.addressableDevices)
}

// DefaultDeviceAssignment for the given number of replicas and partitions.
//
// Replicas refer to data parallelism: the number of identical copies of the program that will be run on
// different data.
//
// Partitions refer to model parallelism: the number of independent copies of the program that will be run on
// different parts of the model. For SPMD programs, this is always 1.
//
// The returned slice is of length numReplicas * numPartitions.
func (c *Client) DefaultDeviceAssignment(numReplicas, numPartitions int) ([]int, error) {
	if c == nil || c.client == nil || !c.client.lifecycle.Enter() {
		return nil, errClientDestroyed("DefaultDeviceAssignment")
	}
	defer c.client.lifecycle.Leave()
	return pjrtClientDefaultDeviceAssignment(c.plugin, c, numReplicas, numPartitions)
}

// NumForDevice returns the "deviceNum" for the given device.
// The value deviceNum is an index to Client.AddressableDevices, and can be used in several other methods.
//
// It returns -1 if the device is not found in Client.AddressableDevices.
func (c *Client) NumForDevice(device *Device) int {
	for deviceNum, otherDevice := range c.addressableDevices {
		if device.localHardwareId == otherDevice.localHardwareId {
			return deviceNum
		}
	}
	return -1
}

// Compile turn a StableHLO program into a "LoadedExecutable" that is the executable runner.
//
// There are different formats of input, and many different compilation options [1], so this returns
// a CompilationConfig that must be furthered configured. At the very least the program must be given: see
// CompileConfig.WithComputation or CompileConfig.WithHLO. Then the call to CompileConfig.Done triggers
// the compilation into a "LoadedExecutable".
//
// [1] The original compilation options is defined as the proto CompileOptionsProto:
// https://github.com/openxla/xla/blob/main/xla/pjrt/compile_options.proto .
// But the proto itself is not documented, instead see documentation in the C++ xla::CompileOptions class defined in:
// https://github.com/openxla/xla/blob/main/xla/pjrt/pjrt_executable.h .
func (c *Client) Compile() *CompileConfig {
	return newCompileConfig(c)
}

// BufferFromHost creates an on-device buffer with the contents copied (optionally reused, if device is CPU) from
// the given host buffer.
//
// It returns a BufferFromHostConfig that must be furthered configured -- at least the host data to transfer must be given.
// Call BufferFromHostConfig.Done to trigger the transfer.
func (c *Client) BufferFromHost() *BufferFromHostConfig {
	return &BufferFromHostConfig{
		client:              c,
		device:              nil,
		hostBufferSemantics: PJRT_HostBufferSemantics_kImmutableUntilTransferCompletes,
	}
}
