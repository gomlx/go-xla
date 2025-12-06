// Package stablehlo helps build a ToStableHLO program (text format) to then be
// JIT-compiled and executed by PJRT (github.com/gomlx/gopjrt/pjrt).
//
// Among its features:
//
// - Translates an API to rendered (human-readable) ToStableHLO text.
// - Shape inference: it calculates the output shapes for operations.
// - Written purely in Go, no C/C++ external dependencies.
//
// It was written as a replacement for `gopjrt/xlabuilder` and attempts to keep
// a similar or identical interface.
//
// See ToStableHLO documentation and specifications in https://openxla.org/stablehlo/spec
package stablehlo

import "github.com/gomlx/go-xla/internal/utils"

// Generates some trivial functions (binary and unary operators) automatically.
//go:generate go run ./internal/cmd/ops_generator

// NormalizeIdentifier converts the name of an identifier (function name or function input parameter
// name, etc.) to a valid one: only letters, digits, and underscores are allowed.
//
// Invalid characters are replaced with underscores.
// If the name starts with a digit, it is prefixed with an underscore.
func NormalizeIdentifier(name string) string {
	return utils.NormalizeIdentifier(name)
}
