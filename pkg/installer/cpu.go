package installer

import (
	"fmt"
	"os"
	"path"
	"path/filepath"
	"strings"

	"github.com/gomlx/go-xla/pkg/pjrt"
	"github.com/pkg/errors"
)

const AmazonLinux = "amazonlinux"

func init() {
	autoInstallers["cpu"] = CPUAutoInstall
}

// CPUAutoInstall installs the latest version of the CPU PJRT if not yet installed.
func CPUAutoInstall(installPath string, useCache bool, verbosity VerbosityLevel) error {
	version := pjrt.DefaultCPUVersion
	pjrtPluginPath := path.Join(installPath, fmt.Sprintf("pjrt_c_api_cpu_%s_plugin.so", version))
	_, err := os.Stat(pjrtPluginPath)
	if err == nil {
		// Already installed.
		return nil
	}
	return CPUInstall("linux", version, installPath, useCache, verbosity)
}

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
func CPUInstall(plugin, version, installPath string, useCache bool, verbosity VerbosityLevel) error {
	// Sequence to clear the line and move to the next line, dependes on verbosity level.
	eolSeq := "\n"
	if verbosity == Normal {
		eolSeq = DeleteToEndOfLine
	}

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
	downloadedFile, inCache, err := DownloadURLToTemp(assetURL, fmt.Sprintf("%s_%s", version, assetName), sha256hash, useCache, verbosity)
	if err != nil {
		return err
	}
	if !inCache {
		defer func() { ReportError(os.Remove(downloadedFile)) }()
	}

	// Extract files
	if verbosity != Quiet {
		fmt.Printf("\r- Extracting files in %s to %s%s", downloadedFile, installPath, eolSeq)
	}
	extractedFiles, err := Untar(downloadedFile, installPath)
	if err != nil {
		return err
	}
	if len(extractedFiles) == 0 {
		return errors.Errorf("failed to extract files from %s", downloadedFile)
	}
	isLinked := false
	if verbosity == Verbose {
		fmt.Printf("- Extracted %d file(s):\n", len(extractedFiles))
	}
	for _, file := range extractedFiles {
		switch verbosity {
		case Verbose:
			fmt.Printf("  - %s\n", file)
		case Normal:
			fmt.Printf("\r- Extracted %d file(s): %s%s", len(extractedFiles), file, DeleteToEndOfLine)
		case Quiet:
		}
		baseFile := filepath.Base(file)
		if !isLinked && strings.HasPrefix(baseFile, "pjrt_c_api_cpu_") && strings.HasSuffix(baseFile, "_plugin.so") {
			// Link file to the default CPU plugin, without the version number.
			linkPath := path.Join(installPath, "pjrt_c_api_cpu_plugin.so")
			if err := os.Remove(linkPath); err != nil && !os.IsNotExist(err) {
				return errors.Wrap(err, "failed to remove existing link")
			}
			if err := os.Symlink(baseFile, linkPath); err != nil {
				return errors.Wrap(err, "failed to create symlink")
			}
			if verbosity == Verbose {
				fmt.Printf("    Linked to %s\n", linkPath)
			}
			isLinked = true
		}
	}
	if verbosity == Verbose {
		fmt.Println()
	}
	if verbosity != Quiet {
		fmt.Printf("\r✅ Installed XLA's PJRT for CPU %s to %s (platform: %s)\n", version, installPath, plugin)
	}
	if verbosity == Verbose {
		fmt.Println()
	}

	return nil
}
