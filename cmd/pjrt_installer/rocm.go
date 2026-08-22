//go:build (linux && amd64) || pjrt_all

package main

import (
	"github.com/gomlx/go-xla/installer"
)

func init() {
	pluginName := "rocm"
	pluginInstallers[pluginName] = func(plugin, version, installPath string) error {
		return installer.RocmInstall(version, installPath, *flagCache, installer.VerbosityLevel(*flagVerbosity))
	}
	pluginValidators[pluginName] = installer.RocmValidateVersion
	pluginValues = append(pluginValues, pluginName)
	pluginDescriptions = append(pluginDescriptions, "ROCm PJRT (linux/amd64, AMD GPU)")
	pluginPriorities = append(pluginPriorities, 15)
}
