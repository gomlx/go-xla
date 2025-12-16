package pool

import (
	"unsafe"
)

// cacheLineSize is the cache line size (64 bytes on amd64).
const cacheLineSize = 64

// maxProcs is the maximum number of processors we support.
// We use a fixed size array to avoid bounds checks and resizing complexity in the hot path.
// 4096 * 64 bytes = 256KB, which is negligible memory overhead.
const maxProcs = 4096

// PoolNode is the node in the linked list.
type PoolNode[T any] struct {
	Item T
	next *PoolNode[T]
}

// poolHead is the head of the linked list for a P.
// It is padded to cacheLineSize to prevent false sharing.
type poolHead[T any] struct {
	head *PoolNode[T]

	// raceMutex is used only when the race detector is enabled to establish
	// happens-before relationships for the race detector. In non-race builds,
	// it is an empty struct and optimized away.
	raceMutex

	// Padding to ensure the struct is 64 bytes.
	// We assume 64-bit architecture (pointer size 8 bytes).
	// 64 - 8 = 56.
	// Note: In race mode, raceMutex adds size, so the total size will be > 64.
	// We disable the size check in race mode.
	_ [56]byte
}

// Pool is a lock-free (per-P) pool of objects.
// Objects in the pool never expire.
type Pool[T any] struct {
	heads []poolHead[T]
	// New optionally specifies a function to generate
	// a value when Get would otherwise return nil.
	// It may not be changed concurrently with calls to Get.
	New func() T
}

// NewPool creates a new Pool.
func NewPool[T any](newFunc func() T) *Pool[T] {
	// Verify alignment assumption.
	if checkAlignment {
		var p poolHead[T]
		if unsafe.Sizeof(p) != cacheLineSize {
			// This should only happen on non-64-bit architectures or if pointer size changes.
			// For now we panic to ensure we meet the spec.
			panic("internal/pool: poolHead size is not 64 bytes")
		}
	}

	return &Pool[T]{
		heads: make([]poolHead[T], maxProcs),
		New:   newFunc,
	}
}

// Get retrieves an object from the pool.
// If the pool is empty, it allocates a new one using New function or zero value.
// Synchronization is done using runtime.procPin/runtime.procUnpin.
func (p *Pool[T]) Get() *PoolNode[T] {
	pid := runtime_procPin()
	// Bounds check.
	if pid >= len(p.heads) {
		runtime_procUnpin()
		// If this happens, maxProcs is insufficient.
		panic("internal/pool: GOMAXPROCS exceeds supported limit")
	}

	h := &p.heads[pid]
	h.lock()
	node := h.head
	if node != nil {
		h.head = node.next
		node.next = nil
	}
	h.unlock()
	runtime_procUnpin()

	if node == nil {
		node = new(PoolNode[T])
		if p.New != nil {
			node.Item = p.New()
		}
	}
	return node
}

// Put returns an object to the pool.
func (p *Pool[T]) Put(node *PoolNode[T]) {
	if node == nil {
		return
	}
	pid := runtime_procPin()
	if pid >= len(p.heads) {
		runtime_procUnpin()
		panic("internal/pool: GOMAXPROCS exceeds supported limit")
	}

	h := &p.heads[pid]
	h.lock()
	node.next = h.head
	h.head = node
	h.unlock()
	runtime_procUnpin()
}
