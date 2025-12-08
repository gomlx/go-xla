//go:build (linux && amd64) || pjrt_all

package main

import (
	"github.com/gomlx/go-xla/pkg/installer"
)

func init() {
	for _, plugin := range []string{"linux", installer.AmazonLinux} {
		pluginInstallers[plugin] = func(plugin, version, installPath string) error {
			return installer.LinuxInstall(plugin, version, installPath, *flagCache)
		}
		pluginValidators[plugin] = installer.LinuxValidateVersion
	}
	pluginValues = append(pluginValues, "linux", installer.AmazonLinux)
	pluginDescriptions = append(pluginDescriptions,
		"CPU PJRT for Linux/amd64 (glibc >= 2.41)",
		"CPU PJRT for AmazonLinux/amd64 and Ubuntu 22 (GCP host systems for TPUs) (glibc >= 2.34)")
	pluginPriorities = append(pluginPriorities, 0, 1)
	installPathSuggestions = append(installPathSuggestions, "~/.local/lib/go-xla", "/usr/local/lib/go-xla")
}
