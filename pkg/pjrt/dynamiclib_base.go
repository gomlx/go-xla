/*
 *	Copyright 2024 Jan Pfeifer
 *
 *	Licensed under the Apache License, Version 2.0 (the "License");
 *	you may not use this file except in compliance with the License.
 *	You may obtain a copy of the License at
 *
 *	http://www.apache.org/licenses/LICENSE-2.0
 *
 *	Unless required by applicable law or agreed to in writing, software
 *	distributed under the License is distributed on an "AS IS" BASIS,
 *	WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *	See the License for the specific language governing permissions and
 *	limitations under the License.
 */

package pjrt

/*
#include <stdlib.h>
#include "pjrt_c_api.h"
#include "common.h"
*/
import "C"
import (
	"os"
	"path/filepath"
	"regexp"
	"slices"
	"strings"
	"sync"

	"github.com/pkg/errors"
	"k8s.io/klog/v2"
)

// This file holds common definitions for the different implementations of dynamiclib (linux, windows, mac?).

const (
	// PJRTPluginPathsEnv is the name of the environment variable that define the search paths for plugins.
	PJRTPluginPathsEnv = "PJRT_PLUGIN_LIBRARY_PATH"

	// GetPJRTApiFunctionName is the name of the function exported by PJRT plugins that returns the API.
	GetPJRTApiFunctionName = "GetPjrtApi"
)

var (
	// pluginSearchPaths is set during initialization by the per-architecture implementations (dynamiclib_<arch>.go files).
	//
	// Plugins are searched in the PJRT_PLUGIN_LIBRARY_PATH directory -- or directories, if it is a ":" separated list.
	// If it is not set it will search in "~/.local/lib/go-xla" and "/usr/local/lib/go-xla" (or equivalent for other OSes),
	// and the standard libraries directories of the system (in linux in LD_LIBRARY_CONFIG and /etc/ld.so.conf file).
	pluginSearchPaths []string

	// loadedPlugins caches the plugins already loaded. Protected by muPlugins.
	loadedPlugins = make(map[string]*Plugin)
	muPlugins     sync.Mutex
)

// dllHandleWrapper encapsulates a handler to the plugin and should provide a minimal interface to get
// the PJRT api function and to close the dll.
//
// It is created with loadPlugin (architecture specific), and one must be able to close it.
type dllHandleWrapper interface {
	// GetPJRTApiFn return C pointer to PJRT API function.
	GetPJRTApiFn() (C.GetPJRTApiFn, error)

	// Close handler, after which the PJRT plugin in no longer valid.
	Close() error
}

func init() {
	pjrtPaths, found := os.LookupEnv(PJRTPluginPathsEnv)
	if !found {
		pluginSearchPaths = osDefaultLibraryPaths()
	} else {
		pluginSearchPaths = slices.DeleteFunc(strings.Split(pjrtPaths, ":"), func(p string) bool {
			return p == "" // Remove empty paths.
		})
	}
}

// loadNamedPlugin by loading the corresponding plugin.
// It returns an error if it doesn't find it.
//
// It uses a mutex to serialize (make it safe) calls from different goroutines.
func loadNamedPlugin(name string) (*Plugin, error) {
	muPlugins.Lock()
	defer muPlugins.Unlock()

	// Search previously loaded plugin: match by name or by path (if the name given is an absolute path).
	if plugin, found := loadedPlugins[name]; found {
		return plugin, nil
	}
	if filepath.IsAbs(name) {
		for _, plugin := range loadedPlugins {
			if plugin.Path() == name {
				return plugin, nil
			}
		}
	}

	// Search path to plugin -- except if name is an absolute path.
	pluginPath := name
	if !filepath.IsAbs(pluginPath) {
		var found bool
		pluginPath, found = searchPlugin(name)
		if !found {
			return nil, errors.Errorf("plugin name %q not found in paths %v: set PJRT_PLUGIN_LIBRARY_PATH to an specific path(s) to search; "+
				"plugins should be named pjrt_c_api_<name>_plugin.so (or .dylib for Darwin, or .dll for Windows)",
				name, pluginSearchPaths)
		}
	}
	klog.V(1).Infof("attempting to load plugin from %s", pluginPath)

	var err error
	handle, err := loadPlugin(pluginPath)
	if err != nil {
		return nil, errors.WithMessagef(err, "failed to load PJRT plugin for name %q", name)
	}
	pjrtAPIFn, err := handle.GetPJRTApiFn()
	if err != nil {
		return nil, errors.WithMessagef(err, "failed to get PJRT plugin API function for name %q", name)
	}
	api := C.call_GetPJRTApiFn(pjrtAPIFn)
	if api == nil {
		return nil, errors.WithMessagef(err, "loaded PJRT plugin for name %q, but it returned a nil plugin!?", name)
	}
	plugin, err := newPlugin(name, pluginPath, api, handle)
	if err != nil {
		return nil, errors.WithMessagef(err, "failed to initialize PJRT plugin for name %q after loading it, this leaves the plugin in an unstable state", name)
	}
	loadedPlugins[name] = plugin
	cudaPluginCheckDrivers(name)
	return plugin, nil
}

var (
	// Patterns to extract the name from the plugins.
	rePluginName = []*regexp.Regexp{
		regexp.MustCompile(`^.*[/\\]pjrt_c_api_(.+)_plugin.(so|dylib|dll)$`),
		regexp.MustCompile(`^.*[/\\]pjrt[-_]plugin[-_](.+).(so|dylib|dll)$`),
	}
)

// pathToPluginName returns the name of the plugin if it's a matching plugin path, otherwise returns "".
func pathToPluginName(pPath string) string {
	for _, re := range rePluginName {
		if re.MatchString(pPath) {
			subMatches := re.FindStringSubmatch(pPath)
			return subMatches[1]
		}
	}
	return ""
}

// AvailablePlugins searches for available plugins in the standard directories and returns a map from their name to their paths.
//
// Plugins are searched in the PJRT_PLUGIN_LIBRARY_PATH directory -- or directories, if it is a ":" separated list.
// If it is not set it will search in the local user "${HOME}/.local/lib/go-xla", the system "/usr/local/lib/go-xla",
// and the standard libraries directories of the system (in linux in LD_LIBRARY_PATH and /etc/ld.so.conf file,
//
//	in Darwin it also searches in DYLD_LIBRARY_PATH) in that order.
//
// (Or equivalent directories for different OSes).
//
// If there are plugins with the same name but different versions in different directories, it respects the order of the
// directories given by PJRT_PLUGIN_LIBRARY_PATH or by the system.
func AvailablePlugins() (pluginsPaths map[string]string) {
	return searchPlugins("")
}

func searchPlugin(searchName string) (path string, found bool) {
	pluginsFound := searchPlugins(searchName)
	path, found = pluginsFound[searchName]
	return path, found
}

func searchPlugins(searchName string) (pluginsPaths map[string]string) {
	pluginsPaths = make(map[string]string)

	// Include plugins already (pre-)loaded.
	for name, pluginPath := range loadedPlugins {
		if searchName != "" && searchName != name {
			continue
		}
		pluginsPaths[name] = pluginPath.Path()
	}

	// Search for plugins in other paths.
	for _, pluginPath := range pluginSearchPaths {
		for _, pattern := range []string{
			"pjrt-plugin-*.so", "pjrt_plugin_*.so", "pjrt_c_api_*_plugin.so",
			"pjrt-plugin-*.dylib", "pjrt_plugin_*.dylib", "pjrt_c_api_*_plugin.dylib",
			"pjrt-plugin-*.dll", "pjrt_plugin_*.dll", "pjrt_c_api_*_plugin.dll"} {
			candidates, err := filepath.Glob(filepath.Join(pluginPath, pattern))
			if err != nil {
				continue
			}
			for _, candidate := range candidates {
				name := pathToPluginName(candidate)
				if name == "" {
					continue
				}
				if searchName != "" && searchName != name {
					continue
				}
				if _, found := pluginsPaths[name]; found {
					// We already have a plugin with that name.
					continue
				}
				err := checkPlugin(name, candidate)
				if err != nil {
					continue
				}
				pluginsPaths[name] = candidate
			}
		}
	}
	return
}

// checkPlugin tries to dlopen the plugin and verify that the GetPjrtApi function is exported.
//
// The handle returned by dlopen is properly destroyed.
func checkPlugin(name, pluginPath string) (err error) {
	if klog.V(1).Enabled() {
		defer func() {
			klog.Infof("Check %q: %v\n", pluginPath, err)
		}()
	}

	if isCuda(name) && !hasNvidiaGPU() {
		return errors.Errorf("plugin %q (%q): no GPU card found, skipping", name, pluginPath)
	}

	var handle dllHandleWrapper
	if err != nil {
		return errors.WithMessagef(err, "failed to load PJRT plugin for %q", pluginPath)
	}
	handle, err = loadPlugin(pluginPath)
	defer func() {
		err2 := handle.Close()
		if err2 != nil {
			klog.Warningf("Failed to close dynamic library %q: %v", pluginPath, err2)
		}
	}()

	var pjrtAPIFn C.GetPJRTApiFn
	pjrtAPIFn, err = handle.GetPJRTApiFn()
	if err != nil {
		err = errors.WithMessagef(err, "failed to get PJRT plugin API function for %q", pluginPath)
		return
	}
	api := C.call_GetPJRTApiFn(pjrtAPIFn)
	if api == nil {
		err = errors.Errorf("loaded PJRT plugin for %q, but it returned a nil plugin!?", pluginPath)
		return
	}
	return
}

// FreeAll frees all plugins resources.
// You should call this only if you are no longer going to use PJRT.
// Useful for debugging memory leaks, this should release the plugins resources -- for these cases
// you should first free buffers, clients and only the plugins.
func FreeAll() {
	muPlugins.Lock()
	defer muPlugins.Unlock()
	for _, p := range loadedPlugins {
		p.Free()
	}
}
