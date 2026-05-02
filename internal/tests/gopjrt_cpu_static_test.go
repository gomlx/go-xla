//go:build pjrt_cpu_static

package tests

import (
	// Link CPU PJRT statically: slower but works on Mac.
	_ "github.com/gomlx/go-xla/pjrt/cpu/static"
)
