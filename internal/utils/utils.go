package utils

// DefaultCPUVersion is the default version of the CPU PJRT plugin to use with
// this go-xla release, against which it was tested.
// Exported in pkg/pjrt, but a copy is kept here to pkg/installer can include it without pulling the whole of PJRT.
var DefaultCPUVersion = "v0.83.3"

// NormalizeIdentifier converts the name of an identifier (function name or function input parameter
// name) to a valid one: only letters, digits, and underscores are allowed.
//
// Invalid characters are replaced with underscores.
// If the name starts with a digit, it is prefixed with an underscore.
func NormalizeIdentifier(name string) string {
	if name == "" {
		return ""
	}
	result := make([]rune, 0, len(name)+1)
	if name[0] >= '0' && name[0] <= '9' {
		result = append(result, '_')
	}
	for _, r := range name {
		if (r >= 'a' && r <= 'z') || (r >= 'A' && r <= 'Z') || (r >= '0' && r <= '9') || r == '_' {
			result = append(result, r)
		} else {
			result = append(result, '_')
		}
	}
	return string(result)
}
