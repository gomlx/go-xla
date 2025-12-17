package installer

import (
	"fmt"
	"os"
	"os/exec"
	"path"
	"path/filepath"
	"regexp"
	"runtime"
	"strconv"
	"strings"

	"github.com/gomlx/go-xla/internal/utils"
	"github.com/pkg/errors"
)

var CPUSupportedPlatforms = []string{
	"linux_amd64", "linux_arm64",
	"darwin_arm64",
	"windows_amd64",
}

const AmazonLinux = "amazonlinux"

func init() {
	autoInstallers["cpu"] = CPUAutoInstall
}

// CPUAutoInstall installs the latest version of the CPU PJRT if not yet installed.
// goxlaInstallPath is expected to be a "lib/go-xla" directory, under which the PJRT plugin is installed.
func CPUAutoInstall(goxlaInstallPath string, useCache bool, verbosity VerbosityLevel) error {
	version := utils.DefaultCPUVersion
	pjrtPluginPath := path.Join(goxlaInstallPath, fmt.Sprintf("pjrt_c_api_cpu_%s_plugin.so", version))
	_, err := os.Stat(pjrtPluginPath)
	if err == nil {
		// Already installed.
		return nil
	}

	platform := fmt.Sprintf("%s_%s", runtime.GOOS, runtime.GOARCH)
	return CPUInstall(platform, version, goxlaInstallPath, useCache, verbosity)
}

var glibcVersionRegex = regexp.MustCompile(`^ldd\s+\(.*\)\s+(\d+)\.(\d+)$`)

// detectGlibcVersion detects the version of the glibc library installed on the system.
func detectGlibcVersion() (major int, minor int, err error) {
	lddBytes, lddErr := exec.Command("ldd", "--version").CombinedOutput()
	if lddErr != nil {
		return 0, 0, errors.Wrap(lddErr, "failed to run ldd --version")
	}
	lines := strings.Split(string(lddBytes), "\n")
	for _, line := range lines {
		matches := glibcVersionRegex.FindStringSubmatch(line)
		if len(matches) == 3 {
			major, _ = strconv.Atoi(matches[1])
			minor, _ = strconv.Atoi(matches[2])
			return major, minor, nil
		}
	}
	return 0, 0, errors.Errorf("glibc version not found in ldd output")
}

// CPUValidateVersion checks whether the linux version selected by "-version" exists.
func CPUValidateVersion(platform, version string) error {
	// "latest" is always valid.
	if version == "latest" {
		return nil
	}

	_, err := CPUGetDownloadURL(platform, version)
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
func CPUGetDownloadURL(platform, version string) (url string, err error) {
	var assets []string
	assets, err = GitHubDownloadReleaseAssets(BinaryCPUReleasesRepo, version)
	if err != nil {
		return "", err
	}
	if len(assets) == 0 {
		return "", errors.Errorf("version %q not found", version)
	}

	extension := ".tar.gz"
	if strings.Contains(platform, "windows") {
		extension = ".zip"
	}
	wantAsset := fmt.Sprintf("pjrt_cpu_%s%s", platform, extension)
	for _, assetURL := range assets {
		if strings.HasSuffix(assetURL, "/"+wantAsset) {
			return assetURL, nil
		}
	}
	return "", errors.Errorf("Plugin %q version %q doesn't seem to have the required asset (%q) -- "+
		"assets found: %v", platform, version, wantAsset, assets)
}

// CPUInstall the assets on the target directory.
func CPUInstall(platform, version, installPath string, useCache bool, verbosity VerbosityLevel) error {
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
	assetURL, err := CPUGetDownloadURL(platform, version)
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
	var extractedFiles []string
	if strings.HasSuffix(downloadedFile, ".zip") {
		extractedFiles, err = Unzip(downloadedFile, installPath)
	} else {
		extractedFiles, err = Untar(downloadedFile, installPath)
	}
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
		if !isLinked {
			if strings.HasPrefix(baseFile, "pjrt_c_api_cpu_") && strings.HasSuffix(baseFile, "_plugin.so") {
				// For Linux/Darwin:
				// Link file to the default CPU plugin, without the version number.
				linkPath := filepath.Join(installPath, "pjrt_c_api_cpu_plugin.so")
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

			} else if strings.HasPrefix(baseFile, "pjrt_c_api_cpu_") && strings.HasSuffix(baseFile, "_plugin.dll") {
				// Windows version of the CPU plugin.
				linkPath := filepath.Join(installPath, "pjrt_c_api_cpu_plugin.dll")
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
	}
	if verbosity == Verbose {
		fmt.Println()
	}
	if verbosity != Quiet {
		fmt.Printf("\r✅ Installed XLA's PJRT for CPU %s to %s (platform: %s)\n", version, installPath, platform)
	}
	if verbosity == Verbose {
		fmt.Println()
	}

	return nil
}
