package pjrt

import (
	"testing"
)

func TestArena(t *testing.T) {
	arena := newArena(1024)
	for range 2 {
		assertEqual(t, 1024, arena.size)
		assertEqual(t, 0, arena.current)
		_ = arenaAlloc[int](arena)
		assertEqual(t, 8, arena.current)
		_ = arenaAlloc[int32](arena)
		assertEqual(t, 16, arena.current)

		_ = arenaAllocSlice[byte](arena, 9) // Aligning, it will occupy 16 bytes total.
		assertEqual(t, 32, arena.current)

		requirePanics(t, func() { _ = arenaAlloc[[512]int](arena) }, "Arena out of memory")
		requirePanics(t, func() { _ = arenaAllocSlice[float64](arena, 512) }, "Arena out of memory")
		arena.Reset()
	}
	arena.Free()
}
