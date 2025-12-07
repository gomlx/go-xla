//go:build (darwin && arm64) || all

package main

import (
	"github.com/gomlx/go-xla/pkg/installer"
)

func init() {
	for _, plugin := range []string{"darwin"} {
		pluginInstallers[plugin] = func(plugin, version, installPath string) error {
			return installer.DarwinInstall(plugin, version, installPath, *flagCache)
		}
		pluginValidators[plugin] = installer.DarwinValidateVersion
	}
	pluginValues = append(pluginValues, "darwin")
	pluginDescriptions = append(pluginDescriptions, "CPU PJRT (darwin/arm64)")
	pluginPriorities = append(pluginPriorities, 3)
	installPathSuggestions = append(installPathSuggestions, "/usr/local/", "~/Library/Application Support/GoMLX")
}
