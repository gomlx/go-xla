//go:build pjrt_cpu_dynamic

package tests

import (
	// Link (preload) CPU PJRT dynamically (as opposed to use `dlopen`).
	_ "github.com/gomlx/go-xla/pjrt/cpu/dynamic"
)
