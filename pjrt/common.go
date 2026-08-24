package pjrt

import "strings"

// This file holds the definition of functions commonly used in different parts.

// keys returns the keys of a map in the form of a slice.
func keys[K comparable, V any](m map[K]V) []K {
	s := make([]K, 0, len(m))
	for k := range m {
		s = append(s, k)
	}
	return s
}

// IsCUDAName tries to guess that the plugin named is associated with Nvidia CUDA, to apply the corresponding hacks.
func IsCUDAName(name string) bool {
	return strings.Contains(strings.ToUpper(name), "CUDA") ||
		strings.Contains(strings.ToUpper(name), "NVIDIA")
}

// IsROCMName tries to guess that the plugin named is associated with AMD ROCm.
func IsROCMName(name string) bool {
	return strings.Contains(strings.ToUpper(name), "ROCM") ||
		strings.Contains(strings.ToUpper(name), "AMD")
}
