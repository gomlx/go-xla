//go:build (linux && amd64) || pjrt_all

package installer

import (
	"fmt"
	"maps"
	"os"
	"path"
	"runtime"
	"slices"
	"strings"

	"sync"

	"github.com/pkg/errors"
	"k8s.io/klog/v2"
)

// HasTPU checks if a TPU is available on the system.
// This uses sync.OnceValue for efficient repeated calls.
var HasTPU = sync.OnceValue(func() bool {
	// Typical file path for libtpu presence. This is a heuristic.
	if _, err := os.Stat("/lib/libtpu.so"); err == nil {
		return true
	}
	if _, err := os.Stat("/usr/lib/libtpu.so"); err == nil {
		return true
	}
	if _, err := os.Stat("/usr/local/lib/libtpu.so"); err == nil {
		return true
	}
	// TPU detection can be expanded as needed.
	return false
})

func init() {
	autoInstallers["tpu"] = TPUAutoInstall
}

const TPUPJRTPluginName = "pjrt_c_api_tpu_plugin.so"

// TPUAutoInstall installs the TPU PJRT if it is available on the system.
// goxlaInstallPath is expected to be a "lib/go-xla" directory, under which the PJRT plugin is installed.
func TPUAutoInstall(goxlaInstallPath string, useCache bool, verbosity VerbosityLevel) (returnErr error) {
	// Only support Linux/amd64 for TPU installation.
	if runtime.GOOS != "linux" {
		return nil
	}
	if !HasTPU() {
		return nil
	}

	pjrtPluginPath := path.Join(goxlaInstallPath, TPUPJRTPluginName)
	isInstalled, fLock, err := checkInstallOrFileLock(pjrtPluginPath)
	if err != nil {
		return err
	}
	if isInstalled {
		return nil
	}

	// We got the lock: makes sure we unlock it at the end and report any errors.
	defer func() {
		errLock := fLock.Unlock()
		if errLock != nil {
			if returnErr == nil {
				returnErr = errLock
			} else {
				// Log the error, continue with the next installer.
				klog.Errorf("AutoInstall error: %+v\n", errLock)
			}
		}
	}()

	// Install it:
	return TPUInstall("tpu", "latest", goxlaInstallPath, useCache, verbosity)
}

// TPUInstall installs the TPU PJRT from the "libtpu" PIP packages, using pypi.org distributed files.
//
// Checks performed:
// - Version exists
// - Downloaded files sha256 match the ones on pypi.org
func TPUInstall(plugin, version, installPath string, useCache bool, verbosity VerbosityLevel) error {
	// Create the target directory.
	var err error
	installPath, err = ReplaceTildeInDir(installPath)
	if err != nil {
		return err
	}
	if err := os.MkdirAll(installPath, 0755); err != nil {
		return errors.Wrapf(err, "failed to create install directory in %s", installPath)
	}
	pjrtOutputPath := path.Join(installPath, TPUPJRTPluginName)

	// Get CUDA PJRT wheel from pypi.org
	info, packageName, err := TPUGetPJRTPipInfo(plugin)
	if err != nil {
		return errors.WithMessagef(err, "can't fetch pypi.org information for %s", plugin)
	}

	// Translate "latest" to the actual version if needed.
	if version == "latest" {
		version = info.Info.Version
	}

	releaseInfos, ok := info.Releases[version]
	if !ok {
		versions := slices.Collect(maps.Keys(info.Releases))
		slices.Sort(versions)
		return errors.Errorf("version %q not found for %q (from pip package %q) -- lastest is %q and existing versions are: %s",
			version, plugin, packageName, info.Info.Version, strings.Join(versions, ", "))
	}

	releaseInfo, err := PipSelectRelease(releaseInfos, PipPackageLinuxAMD64Glibc231(), true)
	if err != nil {
		return errors.Wrapf(err, "failed to find release for %s, version %s", plugin, version)
	}
	if releaseInfo.PackageType != "bdist_wheel" {
		return errors.Errorf("release %s is not a \"binary wheel\" type", releaseInfo.Filename)
	}

	sha256hash := releaseInfo.Digests["sha256"]
	downloadedJaxPJRTWHL, fileCached, err := DownloadURLToTemp(releaseInfo.URL, fmt.Sprintf("gopjrt_%s_%s.whl", packageName, version), sha256hash, useCache, verbosity)
	if err != nil {
		return errors.Wrap(err, "failed to download cuda PJRT wheel")
	}
	if !fileCached {
		defer func() { ReportError(os.Remove(downloadedJaxPJRTWHL)) }()
	}
	pjrtTmpPath := pjrtOutputPath + ".tmp"
	err = ExtractFileFromZip(downloadedJaxPJRTWHL, "libtpu.so", pjrtTmpPath)
	if err != nil {
		_ = os.Remove(pjrtTmpPath)
		return errors.Wrapf(err, "failed to extract TPU PJRT file from %q wheel", packageName)
	}
	if err := os.Rename(pjrtTmpPath, pjrtOutputPath); err != nil {
		_ = os.Remove(pjrtTmpPath)
		return errors.Wrapf(err, "failed to rename %q to %q", pjrtTmpPath, pjrtOutputPath)
	}

	if verbosity == Verbose {
		fmt.Printf("- Installed %s %s to %s\n", plugin, version, pjrtOutputPath)
		fmt.Println()
	}
	if verbosity != Quiet {
		fmt.Printf("\r✅ Installed \"tpu\" PJRT based on PyPI version %s\n", version)
	}
	if verbosity == Verbose {
		fmt.Println()
	}
	return nil
}

// TPUValidateVersion checks whether the TPU version selected by "-version" exists.
func TPUValidateVersion(plugin, version string) error {
	// "latest" is always valid.
	if version == "latest" {
		return nil
	}

	info, packageName, err := TPUGetPJRTPipInfo(plugin)
	if err != nil {
		return errors.WithMessagef(err, "can't fetch pypi.org information for %q", plugin)
	}

	if _, ok := info.Releases[version]; !ok {
		versions := slices.Collect(maps.Keys(info.Releases))
		slices.Sort(versions)
		return errors.Errorf("version %s not found for %s (from pip package %q) -- existing versions: %s",
			version, plugin, packageName, strings.Join(versions, ", "))
	}

	// Version found.
	return nil
}

// TPUGetPJRTPipInfo returns the JSON info for the PIP package that corresponds to the plugin.
func TPUGetPJRTPipInfo(plugin string) (*PipPackageInfo, string, error) {
	var packageName string
	switch plugin {
	case "tpu":
		packageName = "libtpu"
	default:
		return nil, "", errors.Errorf("unknown plugin %q selected", plugin)
	}
	info, err := GetPipInfo(packageName)
	if err != nil {
		return nil, "", errors.Wrapf(err, "failed to get package info for %s", packageName)
	}
	return info, packageName, nil
}
