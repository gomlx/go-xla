//go:build race

package pool

import "sync"

const checkAlignment = false

type raceMutex struct {
	sync.Mutex
}

func (m *raceMutex) lock() {
	m.Lock()
}

func (m *raceMutex) unlock() {
	m.Unlock()
}
