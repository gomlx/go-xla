package installer

import (
	"fmt"
	"os"
	"path"
	"path/filepath"
	"strings"

	"github.com/pkg/errors"
)

const AmazonLinux = "amazonlinux"

// CPUValidateVersion checks whether the linux version selected by "-version" exists.
func CPUValidateVersion(plugin, version string) error {
	// "latest" is always valid.
	if version == "latest" {
		return nil
	}

	_, err := CPUGetDownloadURL(plugin, version)
	if err != nil {
		versions, versionsErr := GitHubGetVersions(BinaryCPUReleasesRepo)
		if versionsErr != nil {
			return errors.WithMessagef(err, "can't fetch PJRT plugin version %q, and I'm not able to "+
				"download list of valid versions -- see "+
				"https://github.com/gomlx/pjrt-cpu-binaries/releases for a list of release versions to choose from",
				version)
		}
		return errors.WithMessagef(err, "can't fetch PJRT plugin version %q, found versions %q", version, versions)
	}
	return nil
}

// CPUGetDownloadURL returns the download URL for the given version and plugin.
func CPUGetDownloadURL(plugin, version string) (url string, err error) {
	var assets []string
	assets, err = GitHubDownloadReleaseAssets(BinaryCPUReleasesRepo, version)
	if err != nil {
		return "", err
	}
	if len(assets) == 0 {
		return "", errors.Errorf("version %q not found", version)
	}

	var wantAsset string
	switch plugin {
	case "linux":
		wantAsset = "pjrt_cpu_linux_amd64.tar.gz"
	case AmazonLinux:
		wantAsset = "pjrt_cpu_amazonlinux_amd64.tar.gz"
	case "darwin":
		wantAsset = "pjrt_cpu_darwin_arm64.tar.gz"
	default:
		return "", errors.Errorf("version validation not implemented for plugin %q in version %s", plugin, version)
	}
	for _, assetURL := range assets {
		if strings.HasSuffix(assetURL, "/"+wantAsset) {
			return assetURL, nil
		}
	}
	return "", errors.Errorf("Plugin %q version %q doesn't seem to have the required asset (%q) -- "+
		"assets found: %v", plugin, version, wantAsset, assets)
}

// CPUInstall the assets on the target directory.
func CPUInstall(plugin, version, installPath string, useCache bool) error {
	var err error
	if version == "latest" || version == "" {
		version, err = GitHubGetLatestVersion()
		if err != nil {
			return err
		}
	}
	assetURL, err := CPUGetDownloadURL(plugin, version)
	if err != nil {
		return err
	}
	assetName := filepath.Base(assetURL)

	// Create the target directory.
	installPath, err = ReplaceTildeInDir(installPath)
	if err != nil {
		return err
	}
	if err := os.MkdirAll(installPath, 0755); err != nil {
		return errors.Wrap(err, "failed to create install directory")
	}

	// Download the asset to a temporary file.
	sha256hash := "" // TODO: no hash for github releases. Is there a way to get them (or get a hardcoded table for all versions?)
	downloadedFile, inCache, err := DownloadURLToTemp(assetURL, fmt.Sprintf("%s_%s", version, assetName), sha256hash, useCache)
	if err != nil {
		return err
	}
	if !inCache {
		defer func() { ReportError(os.Remove(downloadedFile)) }()
	}

	// Extract files
	fmt.Printf("- Extracting files in %s to %s\n", downloadedFile, installPath)
	extractedFiles, err := Untar(downloadedFile, installPath)
	if err != nil {
		return err
	}
	if len(extractedFiles) == 0 {
		return errors.Errorf("failed to extract files from %s", downloadedFile)
	}
	fmt.Printf("- Extracted %d file(s):\n", len(extractedFiles))
	isLinked := false
	for _, file := range extractedFiles {
		fmt.Printf("  - %s\n", file)
		baseFile := filepath.Base(file)
		if !isLinked && strings.HasPrefix(baseFile, "pjrt_c_api_cpu_") && strings.HasSuffix(baseFile, "_plugin.so") {
			// Link file to the default CPU plugin, without the version number.
			linkPath := path.Join(installPath, "pjrt_c_api_cpu_plugin.so")
			if err := os.Remove(linkPath); err != nil && !os.IsNotExist(err) {
				return errors.Wrap(err, "failed to remove existing link")
			}
			if err := os.Symlink(file, linkPath); err != nil {
				return errors.Wrap(err, "failed to create symlink")
			}
			fmt.Printf("    Linked to %s\n", linkPath)
			isLinked = true
		}
	}
	fmt.Println()
	fmt.Printf("✅ Installed XLA's PJRT for CPU %s to %s (platform: %s)", version, installPath, plugin)
	fmt.Println()

	return nil
}
