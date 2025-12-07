//go:build (linux && amd64) || all

package main

import (
	"github.com/gomlx/go-xla/pkg/installer"
)

func init() {
	for _, plugin := range []string{"tpu"} {
		pluginInstallers[plugin] = func(plugin, version, installPath string) error {
			return installer.TPUInstall(plugin, version, installPath, *flagCache)
		}
		pluginValidators[plugin] = installer.TPUValidateVersion
	}
	pluginValues = append(pluginValues, "tpu")
	pluginDescriptions = append(pluginDescriptions,
		"TPU PJRT for Linux/amd64 host machines (glibc >= 2.31)")
	pluginPriorities = append(pluginPriorities, 20)
}
