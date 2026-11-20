// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package pjrt

import (
	"runtime"
	"testing"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/go-xla/stablehlo"
	"github.com/gomlx/go-xla/types/shapes"
)

//go:noinline
func createEphemeralBuffer(t *testing.T, client *Client) {
	t.Helper()
	ephemeral, err := ArrayToBuffer(client, []float32{5, 6, 7, 8}, 4)
	requireNoError(t, err)
	_ = ephemeral
}

func TestClientDestroyDrainsBuffersAndExecutables(t *testing.T) {
	plugin, err := GetPlugin(*FlagPluginName)
	requireNoError(t, err)
	client, err := plugin.NewClient(nil)
	requireNoError(t, err)

	// Stabilize baseline counts after any prior tests' GC cleanups.
	// Note: testing/synctest cannot be used here because runtime.AddCleanup callbacks
	// run outside of any synctest bubble.
	for range 3 {
		runtime.GC()
	}
	baseBuffers := BuffersAlive()
	baseExecs := LoadedExecutablesAlive()

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

	// Create an ephemeral buffer in a non-inlined frame and verify GC collects & unlinks it while client is still alive.
	createEphemeralBuffer(t, client)
	for range 3 {
		runtime.GC()
	}
	assertEqual(t, baseBuffers+2, BuffersAlive())
	assertEqual(t, baseExecs+1, LoadedExecutablesAlive())

	// Destroy the client while exec, inBuf, and outBuf are still strongly referenced!
	requireNoError(t, client.Destroy())
	if client.IsValid() {
		t.Fatalf("client should be invalid after Destroy()")
	}

	// Both tracked buffers and the executable must have been destroyed immediately, before GC.
	assertEqual(t, baseBuffers, BuffersAlive())
	assertEqual(t, baseExecs, LoadedExecutablesAlive())

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

	// Subsequent explicit Destroy calls and GC must be safe no-ops and not underflow counters.
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

	assertEqual(t, baseBuffers, BuffersAlive())
	assertEqual(t, baseExecs, LoadedExecutablesAlive())
}
