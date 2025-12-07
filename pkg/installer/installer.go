// Package installer provides functionality to install PJRT plugins.
//
// The API exposes several functions to install the different PJRT plugins individually, and it is used
// the command-line program github.com/gomlx/go-xla/cmd/pjrt_installer.
//
// External users may be interested in the using the AutoInstall function to automatically install the PRJT
// plugin for the current platform.
//
// By default, the functions are only available to the corresponding platforms (currently only
// Linux/amd64 and Darwin/arm64). If you use the tag `pjrt_all`, all functions will be available.
// The AutoInstall function is an exception, it is platform-specific and the tag `pjrt_all` has no effect on it.
package installer
