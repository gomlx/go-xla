//go:build !race_mu

package pool

type raceMutex struct {
	_ [8]byte
}

func (m *raceMutex) lock()   {}
func (m *raceMutex) unlock() {}
