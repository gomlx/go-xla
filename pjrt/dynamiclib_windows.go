//go:build windows

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

// This file handles management of loading dynamic libraries for Windows.
//
// It should implement 2 methods:
//
//	osDefaultLibraryPaths() []string
//	loadPlugin(path string) (handleWrapper dllHandleWrapper, err error)

/*
#include <stdlib.h>
#include "pjrt_c_api.h"
#include "common.h"
*/
import "C"
import (
	"os"
	"path/filepath"
	"syscall"
	"unsafe"

	"github.com/pkg/errors"
	"k8s.io/klog/v2"
)

// osDefaultLibraryPaths is called during initialization to set the default search paths.
func osDefaultLibraryPaths() []string {
	var paths []string
	homeDir, err := os.UserHomeDir()
	if err != nil {
		klog.Errorf("Couldn't get user's home directory: %v", err)
		return paths
	}
	// User specified: ...\AppData\Local\go-xla
	paths = append(paths, filepath.Join(homeDir, "AppData", "Local", "go-xla"))
	return paths
}

// loadPlugin tries to load the plugin and returns a handle with the pointer to the PJRT api function
func loadPlugin(pluginPath string) (handleWrapper dllHandleWrapper, err error) {
	klog.V(2).Infof("trying to load library %s\n", pluginPath)

	// syscall.LoadLibrary takes string name.
	handle, err := syscall.LoadLibrary(pluginPath)
	if err != nil {
		err = errors.Wrapf(err, "failed to load PJRT plugin from %q", pluginPath)
		klog.Warningf("%v", err)
		return nil, err
	}

	klog.V(1).Infof("loaded library %s\n", pluginPath)

	// Get Proc Address
	procAddr, err := syscall.GetProcAddress(handle, GetPJRTApiFunctionName)
	if err != nil {
		syscall.FreeLibrary(handle)
		err = errors.Wrapf(err, "failed to find symbol %q in %q", GetPJRTApiFunctionName, pluginPath)
		klog.Warningf("%v", err)
		return nil, err
	}

	h := &windowsDLLHandle{
		Handle:    handle,
		Name:      pluginPath,
		PJRTApiFn: (C.GetPJRTApiFn)(unsafe.Pointer(procAddr)),
	}
	return h, nil
}

// windowsDLLHandle represents an open handle to a library (.dll)
type windowsDLLHandle struct {
	Handle    syscall.Handle
	PJRTApiFn C.GetPJRTApiFn
	Name      string
}

// GetPJRTApiFn returns the pointer to the PJRT API function.
func (h *windowsDLLHandle) GetPJRTApiFn() (C.GetPJRTApiFn, error) {
	return h.PJRTApiFn, nil
}

// Close closes a LibHandle.
func (h *windowsDLLHandle) Close() error {
	err := syscall.FreeLibrary(h.Handle)
	if err != nil {
		return errors.Wrapf(err, "error closing %v", h.Name)
	}
	return nil
}

// SuppressAbseilLoggingHack is a placeholder for Windows.
//
// On Linux/Posix it suppresses stderr, which is used by Abseil logging.
// Implementation on Windows would be different, and for now we leave it as a no-op.
func SuppressAbseilLoggingHack(fn func()) {
	// TODO: Implement suppression of stderr for Windows if needed/possible.
	fn()
}
