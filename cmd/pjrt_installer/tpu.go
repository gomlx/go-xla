//go:build (linux && amd64) || pjrt_all

package main

import (
	"github.com/gomlx/go-xla/pkg/installer"
)

func init() {
	pluginName := "tpu"
	pluginInstallers[pluginName] = func(plugin, version, installPath string) error {
		return installer.TPUInstall(plugin, version, installPath, *flagCache, installer.VerbosityLevel(*flagVerbosity))
	}
	pluginValidators[pluginName] = installer.TPUValidateVersion
	pluginValues = append(pluginValues, pluginName)
	pluginDescriptions = append(pluginDescriptions, "TPU PJRT (linux/amd64)")
	pluginPriorities = append(pluginPriorities, 20)
}
