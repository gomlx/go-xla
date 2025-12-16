//go:build !race

package pool

const checkAlignment = true

type raceMutex struct{}

func (m *raceMutex) lock() {}
func (m *raceMutex) unlock() {}
