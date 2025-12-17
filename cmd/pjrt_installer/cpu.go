package main

import (
	"fmt"
	"path/filepath"
	"runtime"
	"slices"

	"github.com/gomlx/go-xla/pkg/installer"
	"k8s.io/klog/v2"
)

func init() {
	platform := fmt.Sprintf("%s_%s", runtime.GOOS, runtime.GOARCH)
	if !slices.Contains(installer.CPUSupportedPlatforms, platform) {
		return
	}
	pluginName := "cpu"
	pluginInstallers[pluginName] = func(plugin, version, installPath string) error {
		// Platform set to current architecture.
		return installer.CPUInstall(platform, version, installPath, *flagCache, installer.VerbosityLevel(*flagVerbosity))
	}
	pluginValidators[pluginName] = func(plugin, version string) error {
		return installer.CPUValidateVersion(platform, version)
	}
	pluginValues = append(pluginValues, pluginName)
	pluginDescriptions = append(pluginDescriptions, fmt.Sprintf("CPU PJRT (%s/%s)", runtime.GOOS, runtime.GOARCH))
	pluginPriorities = append(pluginPriorities, 0)

	// Install path suggestions.
	localLibPath, err := installer.DefaultHomeLibPath()
	if err != nil {
		klog.Fatalf("Failed to find home directory: %+v", localLibPath)
	}
	installPathSuggestions = append(installPathSuggestions, filepath.Join(localLibPath, "go-xla"))
	if slices.Contains([]string{"darwin", "linux"}, runtime.GOOS) {
		installPathSuggestions = append(installPathSuggestions, "/usr/local/lib/go-xla")
	}
}
