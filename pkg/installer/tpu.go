//go:build (linux && amd64) || pjrt_all

package installer

import (
	"fmt"
	"maps"
	"os"
	"path"
	"slices"
	"strings"

	"github.com/pkg/errors"
)

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
	pjrtOutputPath := path.Join(installPath, "pjrt_c_api_tpu_plugin.so")

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
	err = ExtractFileFromZip(downloadedJaxPJRTWHL, "libtpu.so", pjrtOutputPath)
	if err != nil {
		return errors.Wrapf(err, "failed to extract TPU PJRT file from %q wheel", packageName)
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
