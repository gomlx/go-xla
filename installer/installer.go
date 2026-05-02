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

import (
	"os"
	"path/filepath"
	"runtime"

	"github.com/pkg/errors"
	"k8s.io/klog/v2"
)

// AutoInstaller is a function that automatically installs the PJRT plugin for the current platform.
// It should be safe to call multiple times, and be a no-op if the plugin is already installed.
//
// It is expected to only be called concurrently.
type AutoInstaller func(installPath string, useCache bool, verbosity VerbosityLevel) error

var (
	// autoInstallers is a list of functions that automatically install a PJRT plugin for the current platform,
	// if the corresponding hardware is present.
	autoInstallers = make(map[string]AutoInstaller)
)

// AutoInstall automatically installs the PJRT plugin for the current platform,
// if the corresponding hardware is present.
//
// It returns the first error encountered, or nil if no error occurred.
//
//   - installPath should point to a "lib" directory, under which an "go-xla" subdirectory is created,
//     and the PJRT plugins are installed (e.g.: "~/.local/lib", "/usr/local/lib/").
//     If it is set to "", it is set to the default user-local library directory (
//     in Linux is $HOME/.local/lib/, and on MacOS is $HOME/Library/Application Support/).
//
// - useCache: if true, it will use the cache to store downloaded files. Recommended to keep it true.
//
// - verbosity: the verbosity level to use. 0=quiet, 1=normal (1 log line per plugin installed), 2=verbose.
func AutoInstall(installPath string, useCache bool, verbosity VerbosityLevel) error {
	if installPath == "" {
		var err error
		installPath, err = DefaultHomeLibPath()
		if err != nil {
			return err
		}
	}
	goxlaInstallPath := filepath.Join(installPath, "go-xla")
	var firstErr error
	for installerName, installer := range autoInstallers {
		if err := installer(goxlaInstallPath, useCache, verbosity); err != nil {
			err = errors.WithMessagef(err, "failed to auto-install %q", installerName)
			if firstErr == nil {
				firstErr = err
			} else {
				// Log the error, continue with the next installer.
				klog.Errorf("AutoInstall error: %+v\n", err)
			}
		}
	}
	return firstErr
}

// DefaultHomeLibPath returns the default user-local library directory ("~/.local/lib" in Linux)
//
// This is the directory used by AutoInstall if the installPath is empty.
func DefaultHomeLibPath() (string, error) {
	homeDir, err := os.UserHomeDir()
	if err != nil {
		return "", err
	}
	switch runtime.GOOS {
	case "linux":
		return filepath.Join(homeDir, ".local", "lib"), nil
	case "darwin":
		return filepath.Join(homeDir, "Library", "Application Support"), nil
	case "windows":
		return filepath.Join(homeDir, "AppData", "Local"), nil
	default:
		return "", errors.Errorf("auto-install not supported on platform %s/%s", runtime.GOOS, runtime.GOARCH)
	}
}
