package main

import (
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"strings"

	"github.com/gomlx/go-xla/internal/utils"
	"github.com/gomlx/go-xla/pkg/installer"
	"github.com/pkg/errors"
	"k8s.io/klog/v2"
)

var (
	pluginValues           []string
	pluginDescriptions     []string
	pluginPriorities       []int // Order to display the plugins: smaller values are displayed first.
	pluginInstallers       = make(map[string]func(plugin, version, installPath string) error)
	pluginValidators       = make(map[string]func(plugin, version string) error)
	installPathSuggestions []string

	flagPlugin, flagPath *string
	flagVersion          = flag.String("version", "latest",
		"For PJRT for CPUs, this is the https://github.com/gomlx/pjrt-cpu-binaries release version (e.g.: v0.83.1) "+
			"from where to download the plugin. "+
			"For the CUDA PJRT this is based on the Jax version in https://pypi.org/project/jax/ (e.g.: 0.7.2), "+
			"which is where it downloads the plugin and Nvidia libraries from. "+
			"For the TPU PJRT this is the version of the \"libtpu\" version in https://pypi.org/project/libtpu/ "+
			"(e.g.: 0.0.27). ")
	flagCache = flag.Bool("cache", true, "Use cache to store downloaded files. It defaults to true")
)

func main() {
	if len(pluginValues) == 0 {
		klog.Fatalf("no installable plugins registered for platform %s/%s", runtime.GOOS, runtime.GOARCH)
	}

	// Initialize and set default values for flags
	// klog.InitFlags(nil)

	// Sort plugins by priority
	for i := 0; i < len(pluginPriorities); i++ {
		for j := i + 1; j < len(pluginPriorities); j++ {
			if pluginPriorities[i] > pluginPriorities[j] {
				pluginPriorities[i], pluginPriorities[j] = pluginPriorities[j], pluginPriorities[i]
				pluginValues[i], pluginValues[j] = pluginValues[j], pluginValues[i]
				pluginDescriptions[i], pluginDescriptions[j] = pluginDescriptions[j], pluginDescriptions[i]
			}
		}
	}

	// Make installPathSuggestions unique while preserving the order:
	seen := utils.MakeSet[string](len(installPathSuggestions))
	writeIdx := 0
	for _, path := range installPathSuggestions {
		if !seen.Has(path) {
			seen.Insert(path)
			installPathSuggestions[writeIdx] = path
			writeIdx++
		}
	}
	installPathSuggestions = installPathSuggestions[:writeIdx]

	// Define flags with plugins configured for GOOS/GOARCH used to build this binary:
	flagPlugin = flag.String("plugin", "", "Plugin to install. Valid values: "+strings.Join(pluginValues, ", "))
	flagPath = flag.String("path", "",
		fmt.Sprintf("Installation base path, under which the required libraries and include files are installed. "+
			"For CUDA plugins, it also creates a subdirectory 'nvidia/' to install "+
			"Nvidia's matching libraries/drivers. Suggestions: %s. "+
			"It will require the adequate privileges (sudo) if installing in a system directories.",
			strings.Join(installPathSuggestions, ", ")))

	// Parse flags.
	flag.Parse()

	if *flagPlugin == "" || *flagPath == "" || *flagVersion == "" {
		questions := []Question{
			{Title: "Plugin to install", Flag: flag.CommandLine.Lookup("plugin"),
				Values: pluginValues, ValuesDescriptions: pluginDescriptions, CustomValues: false},
			{Title: "Plugin version", Flag: flag.CommandLine.Lookup("version"), Values: []string{"latest"}, CustomValues: true,
				ValidateFn: ValidateVersion},
			{Title: "Path where to install", Flag: flag.CommandLine.Lookup("path"), Values: installPathSuggestions, CustomValues: true,
				ValidateFn: ValidatePathPermission},
		}
		err := Interact(os.Args[0], questions)
		if err != nil {
			if err == ErrUserAborted {
				fmt.Println("Installation aborted.")
				return
			}
			klog.Fatalf("Failed on error: %+v", err)
		}
	}

	pluginName := *flagPlugin
	version := *flagVersion
	installPath, err := installer.ReplaceTildeInDir(*flagPath)
	if err != nil {
		klog.Fatalf("Failed on error: %+v", err)
	}
	fmt.Printf("Installing PJRT plugin %s@%s to %s:\n", pluginName, version, installPath)

	pluginInstaller, ok := pluginInstallers[pluginName]
	if !ok {
		klog.Fatalf("Installer for plugin %q not found", pluginName)
	}
	if err := pluginInstaller(pluginName, version, installPath); err != nil {
		klog.Fatalf("Failed on error: %+v", err)
	}
}

// ValidateVersion is called to validate the version of the plugin chosen by the user during the interactive mode.
func ValidateVersion() error {
	validator, ok := pluginValidators[*flagPlugin]
	if !ok {
		return errors.Errorf("version validation not implemented for plugin %q", *flagPlugin)
	}
	return validator(*flagPlugin, *flagVersion)
}

func ValidatePathPermission() error {
	installPath, err := installer.ReplaceTildeInDir(*flagPath)
	if err != nil {
		return err
	}
	dir := installPath
	_, err = os.Stat(dir)
	if err != nil {
		// If the directory doesn't exist, try parent directories
		parent := installPath
		for {
			parent = filepath.Dir(parent)
			if parent == "/" || parent == "." {
				return errors.New("could not find an existing parent directory")
			}
			if _, err := os.Stat(parent); err == nil {
				dir = parent
				break
			}
		}
	}

	// Try to create a temporary file to verify write permissions
	testFile := filepath.Join(dir, ".gopjrt_write_test")
	f, err := os.Create(testFile)
	if err != nil {
		return errors.Wrapf(err, "no write permission in directory %q, do you need \"sudo\" ?", dir)
	}
	installer.ReportError(f.Close())

	// Clean up test file
	installer.ReportError(os.Remove(testFile))

	return nil
}
