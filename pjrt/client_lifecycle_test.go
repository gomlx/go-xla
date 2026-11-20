// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package pjrt

import (
	"runtime"
	"testing"
	"weak"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/go-xla/stablehlo"
	"github.com/gomlx/go-xla/types/shapes"
)

//go:noinline
func createEphemeralBuffer(t *testing.T, client *Client) weak.Pointer[Buffer] {
	t.Helper()
	ephemeral, err := ArrayToBuffer(client, []float32{5, 6, 7, 8}, 4)
	requireNoError(t, err)
	return weak.Make(ephemeral)
}

func TestClientDestroyDrainsBuffersAndExecutables(t *testing.T) {
	plugin, err := GetPlugin(*FlagPluginName)
	requireNoError(t, err)
	client, err := plugin.NewClient(nil)
	requireNoError(t, err)

	assertEqual(t, 0, client.client.numTrackedBuffers())
	assertEqual(t, 0, client.client.numTrackedExecutables())

	// Compile a tiny program x + x.
	b := stablehlo.New("lifecycle_test")
	fn := b.Main()
	x, err := fn.NamedInput("x", shapes.Make(dtypes.Float32, 4))
	requireNoError(t, err)
	y, err := stablehlo.Add(x, x)
	requireNoError(t, err)
	requireNoError(t, fn.Return(y))
	program, err := b.Build()
	requireNoError(t, err)

	exec, err := client.Compile().WithStableHLO(program).Done()
	requireNoError(t, err)

	// Create buffers and leave them strongly referenced (not explicitly Destroyed).
	inBuf, err := ArrayToBuffer(client, []float32{1, 2, 3, 4}, 4)
	requireNoError(t, err)
	outs, err := exec.Execute(inBuf).Done()
	requireNoError(t, err)
	outBuf := outs[0]

	// Create an ephemeral buffer in a non-inlined frame and verify GC collects it
	// (proving clientC's intrusive *bufferWrapper list does not retain *Buffer) and that
	// its runtime.AddCleanup unlinks the wrapper while the client is still alive.
	// Note: testing/synctest cannot be used here because runtime.AddCleanup callbacks
	// run outside of any synctest bubble.
	ephemeralWeak := createEphemeralBuffer(t, client)
	for range 3 {
		runtime.GC()
	}
	if ephemeralWeak.Value() != nil {
		t.Fatalf("ephemeral Buffer was not garbage-collected; Client tracking must not retain strong references to *Buffer")
	}
	// runtime.AddCleanup runs CGo PJRT_Buffer_Destroy asynchronously on a background
	// cleanup goroutine (potentially queued behind cleanups from earlier tests in package pjrt).
	for client.client.numTrackedBuffers() > 2 {
		runtime.Gosched()
	}
	assertEqual(t, 2, client.client.numTrackedBuffers())
	assertEqual(t, 1, client.client.numTrackedExecutables())

	// Destroy the client while exec, inBuf, and outBuf are still strongly referenced!
	requireNoError(t, client.Destroy())
	if client.IsValid() {
		t.Fatalf("client should be invalid after Destroy()")
	}

	// Both tracked buffers and the executable must have been destroyed immediately, before GC.
	assertEqual(t, 0, client.client.numTrackedBuffers())
	assertEqual(t, 0, client.client.numTrackedExecutables())
	assertFalse(t, inBuf.wrapper.IsValid())
	assertFalse(t, outBuf.wrapper.IsValid())
	assertFalse(t, exec.wrapper.isValid())

	// Using stale objects after Client.Destroy must return ErrBackendFinalized without crashing.
	if _, err := exec.Execute(inBuf).Done(); !compute.IsBackendFinalized(err) {
		t.Errorf("exec.Execute after Client.Destroy: want ErrBackendFinalized, got %v", err)
	}
	if err := inBuf.ToHost(make([]byte, 16)); !compute.IsBackendFinalized(err) {
		t.Errorf("inBuf.ToHost after Client.Destroy: want ErrBackendFinalized, got %v", err)
	}
	if err := outBuf.ToHost(make([]byte, 16)); !compute.IsBackendFinalized(err) {
		t.Errorf("outBuf.ToHost after Client.Destroy: want ErrBackendFinalized, got %v", err)
	}
	if _, err := client.Compile().WithStableHLO(program).Done(); !compute.IsBackendFinalized(err) {
		t.Errorf("client.Compile after Client.Destroy: want ErrBackendFinalized, got %v", err)
	}

	// Subsequent explicit Destroy calls and GC must be safe no-ops.
	requireNoError(t, inBuf.Destroy())
	requireNoError(t, outBuf.Destroy())
	requireNoError(t, exec.Destroy())
	requireNoError(t, client.Destroy())

	runtime.KeepAlive(inBuf)
	runtime.KeepAlive(outBuf)
	runtime.KeepAlive(exec)
	for range 3 {
		runtime.GC()
	}

	assertEqual(t, 0, client.client.numTrackedBuffers())
	assertEqual(t, 0, client.client.numTrackedExecutables())
}
