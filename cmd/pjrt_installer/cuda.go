//go:build (linux && amd64) || pjrt_all

package main

import (
	"github.com/gomlx/go-xla/pkg/installer"
)

func init() {
	for _, plugin := range []string{"cuda13", "cuda12"} {
		pluginInstallers[plugin] = func(plugin, version, installPath string) error {
			return installer.CudaInstall(plugin, version, installPath, *flagCache, installer.VerbosityLevel(*flagVerbosity))
		}
		pluginValidators[plugin] = installer.CudaValidateVersion
	}
	pluginValues = append(pluginValues, "cuda13", "cuda12")
	pluginDescriptions = append(pluginDescriptions,
		"CUDA PJRT (linux/amd64), using CUDA 13",
		"CUDA PJRT (linux/amd64), using CUDA 12, deprecated")
	pluginPriorities = append(pluginPriorities, 10, 11)
}
