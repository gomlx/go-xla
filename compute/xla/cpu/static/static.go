// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

// Package static links the XLA/PJRT CPU plugin statically with your binary.
//
// This is slower than dynamically (pre-)linking, but it may be convenient because the binary won't dependent
// on other files to run -- except the standard C/C++ libraries, but those are usually available in most boxes.
//
// To use it, import it:
//
//	import _ "github.com/gomlx/go-xla/compute/xla/cpu/static"
//
// It also automatically includes the XLA engine ("github.com/gomlx/go-xla/compute/xla").
//
// See also github.com/gomlx/go-xla/compute/xla/cpu/dynamic for dynamic linking.
package static

import (
	// Link XLA engine.
	_ "github.com/gomlx/go-xla/compute/xla"

	// Link CPU PJRT statically.
	_ "github.com/gomlx/go-xla/pjrt/cpu/static"
)
