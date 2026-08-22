//go:build (linux && amd64) || pjrt_all

package installer

import (
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"os/exec"
	"path"
	"path/filepath"
	"regexp"
	"runtime"
	"strings"
	"sync"

	"github.com/pkg/errors"
	"k8s.io/klog/v2"
)

const ROCMPJRTPluginFileName = "pjrt_c_api_rocm_plugin.so"

// ROCmBaseURL is the base URL of AMD's manylinux repository, which hosts the
// ROCm JAX wheels (including the ROCm PJRT plugin) under `rocm-rel-<version>/`.
const ROCmBaseURL = "https://repo.radeon.com/rocm/manylinux"

// HasAMDGPU tries to guess if there is an actual discrete AMD GPU with ROCm
// installed. Integrated (APU) GPUs share system memory and are not suitable
// for ML workloads, so they are ignored.
//
// It checks for the presence of the /dev/kfd device file (the AMD ROCm compute
// device) and then parses `rocminfo` to tell a discrete GPU apart from an APU.
var HasAMDGPU = sync.OnceValue[bool](func() bool {
	if _, err := os.Stat("/dev/kfd"); err != nil {
		return false
	}
	output := runRocminfo()
	if output == "" {
		// Can't run rocminfo; assume the AMD GPU is usable.
		return true
	}
	return rocminfoHasDiscreteGPU(output)
})

// rocmInstallDir returns the ROCm installation directory. It is used to locate
// `rocminfo` and the ROCm version file when ROCm is not installed in the
// default /opt/rocm location.
//
// It checks, in order:
//  1. The ROCM_PATH environment variable, if set to an existing directory.
//  2. The install root inferred from the `rocminfo` binary found in PATH
//     (its parent directory's parent, e.g. <root>/bin/rocminfo -> <root>).
//  3. The default /opt/rocm.
func rocmInstallDir() string {
	if dir := os.Getenv("ROCM_PATH"); dir != "" {
		if info, err := os.Stat(dir); err == nil && info.IsDir() {
			return dir
		}
	}
	if p, err := exec.LookPath("rocminfo"); err == nil {
		if real, err := filepath.EvalSymlinks(p); err == nil {
			p = real
		}
		return filepath.Dir(filepath.Dir(p))
	}
	return "/opt/rocm"
}

// runRocminfo executes `rocminfo` and returns its output, or "" if it is not
// available.
func runRocminfo() string {
	var cmdPath string
	if p, err := exec.LookPath("rocminfo"); err == nil {
		cmdPath = p
	} else {
		candidate := filepath.Join(rocmInstallDir(), "bin", "rocminfo")
		if _, err := os.Stat(candidate); err == nil {
			cmdPath = candidate
		} else {
			klog.V(1).Infof("rocminfo not found; assuming any AMD GPU is a discrete GPU")
			return ""
		}
	}
	cmd := exec.Command(cmdPath)
	output, err := cmd.CombinedOutput()
	if err != nil {
		klog.V(1).Infof("rocminfo failed to execute: %v", err)
		return ""
	}
	return string(output)
}

// rocminfoHasDiscreteGPU parses `rocminfo` output and reports whether at least
// one AMD GPU agent is a discrete GPU (as opposed to an integrated APU, which
// reports "Memory Properties: APU").
func rocminfoHasDiscreteGPU(output string) bool {
	blocks := strings.Split(output, "*******")
	for _, block := range blocks {
		name := rocminfoField(block, "Name:")
		if !strings.HasPrefix(name, "gfx") {
			continue // Not a GPU agent (e.g. the CPU agent).
		}
		if rocminfoField(block, "Memory Properties:") != "APU" {
			return true
		}
	}
	return false
}

// rocminfoField returns the value of the given field (e.g. "Name:") within a
// rocminfo agent block, or "" if not found.
func rocminfoField(block, field string) string {
	idx := strings.Index(block, field)
	if idx < 0 {
		return ""
	}
	rest := block[idx+len(field):]
	if nl := strings.IndexByte(rest, '\n'); nl >= 0 {
		rest = rest[:nl]
	}
	return strings.TrimSpace(rest)
}

func init() {
	autoInstallers["rocm"] = RocmAutoInstall
}

// RocmAutoInstall installs the ROCm PJRT plugin if not yet installed, and there
// is an actual AMD GPU with ROCm installed. It auto-detects the ROCm version
// from the system.
//
// goxlaInstallPath is expected to be a "lib/go-xla" directory, under which the
// ROCm PJRT plugin is installed.
func RocmAutoInstall(goxlaInstallPath string, useCache bool, verbosity VerbosityLevel) (returnErr error) {
	if runtime.GOOS != "linux" || runtime.GOARCH != "amd64" {
		// Only supported on Linux/amd64.
		return nil
	}
	if !HasAMDGPU() {
		// No need to install anything.
		return nil
	}

	pjrtPluginPath := path.Join(goxlaInstallPath, ROCMPJRTPluginFileName)
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

	version, err := RocmDetectedVersion()
	if err != nil {
		return err
	}
	return RocmInstall(version, goxlaInstallPath, useCache, verbosity)
}

// RocmInstall installs the ROCm PJRT plugin from AMD's manylinux repository,
// matching the given ROCm version (e.g. "7.2.4"). If version is "latest" or
// empty, the installed ROCm version is auto-detected.
//
// Unlike the CUDA plugin, the ROCm PJRT links against the system ROCm libraries
// (its RUNPATH includes /opt/rocm/lib), so no additional libraries need to be
// installed alongside the plugin.
//
// The installPath parameter should be the .../lib/go-xla directory.
func RocmInstall(version, installPath string, useCache bool, verbosity VerbosityLevel) error {
	if version == "latest" || version == "" {
		var err error
		version, err = RocmDetectedVersion()
		if err != nil {
			return err
		}
	}

	var err error
	installPath, err = ReplaceTildeInDir(installPath)
	if err != nil {
		return err
	}
	if err := os.MkdirAll(installPath, 0755); err != nil {
		return errors.Wrapf(err, "failed to create install directory in %s", installPath)
	}
	pjrtOutputPath := path.Join(installPath, ROCMPJRTPluginFileName)

	wheelURL, err := RocmGetWheelURL(version)
	if err != nil {
		return err
	}

	// The AMD repository does not publish SHA256 digests for its wheels, so the
	// downloaded file is not hash-verified (unlike the pypi.org distributions).
	downloadedWHL, fileCached, err := DownloadURLToTemp(
		wheelURL, fmt.Sprintf("go-xla_jax_rocm_pjrt_rocm%s.whl", version), "", useCache, verbosity)
	if err != nil {
		return errors.Wrap(err, "failed to download ROCm PJRT wheel")
	}
	if !fileCached {
		defer func() { ReportError(os.Remove(downloadedWHL)) }()
	}

	pjrtTmpPath := pjrtOutputPath + ".tmp"
	err = ExtractFileFromZip(downloadedWHL, "xla_rocm_plugin.so", pjrtTmpPath)
	if err != nil {
		_ = os.Remove(pjrtTmpPath)
		return errors.Wrapf(err, "failed to extract ROCm PJRT file from wheel %q", downloadedWHL)
	}
	if err := os.Rename(pjrtTmpPath, pjrtOutputPath); err != nil {
		_ = os.Remove(pjrtTmpPath)
		return errors.Wrapf(err, "failed to rename %q to %q", pjrtTmpPath, pjrtOutputPath)
	}
	switch verbosity {
	case Verbose:
		fmt.Printf("- Installed ROCm %s PJRT to %s\n", version, pjrtOutputPath)
	case Normal:
		fmt.Printf("\r- Installed ROCm %s PJRT to %s%s", version, pjrtOutputPath, DeleteToEndOfLine)
	case Quiet:
	}
	return nil
}

// RocmDetectedVersion returns the installed ROCm version (e.g. "7.2.4"), read
// from the ROCm version file, located using rocmInstallDir.
func RocmDetectedVersion() (string, error) {
	root := rocmInstallDir()
	for _, p := range []string{
		filepath.Join(root, ".info", "version"),
		filepath.Join(root, "lib", ".info", "version"),
	} {
		b, err := os.ReadFile(p)
		if err != nil {
			continue
		}
		version := strings.TrimSpace(string(b))
		if version != "" {
			return version, nil
		}
	}
	return "", errors.Errorf("could not detect the installed ROCm version: "+
		"no version file found under %q", root)
}

// RocmGetWheelURL returns the download URL of the ROCm PJRT wheel for the given
// ROCm version (e.g. "7.2.4"). It fetches the directory listing of AMD's
// manylinux repository and selects the most compatible wheel.
func RocmGetWheelURL(version string) (string, error) {
	dirURL := fmt.Sprintf("%s/rocm-rel-%s/", ROCmBaseURL, version)
	hrefs, err := fetchHTMLHrefs(dirURL)
	if err != nil {
		return "", errors.Wrapf(err, "failed to list ROCm %q wheels", version)
	}
	wheel := rocmSelectWheel(hrefs)
	if wheel == "" {
		return "", errors.Errorf("no ROCm PJRT wheel found for version %q at %s -- is that a valid ROCm version?", version, dirURL)
	}
	return dirURL + wheel, nil
}

// RocmValidateVersion checks whether the ROCm version selected by "-version"
// exists in AMD's manylinux repository.
func RocmValidateVersion(plugin, version string) error {
	if version == "latest" {
		return nil
	}
	_, err := RocmGetWheelURL(version)
	return err
}

// rocmSelectWheel picks the most compatible ROCm PJRT wheel from a list of
// (URL-encoded) hrefs, preferring manylinux_2_28 > manylinux2014 > other
// manylinux > linux_x86_64. It returns the raw href, or "" if no PJRT wheel is
// found.
func rocmSelectWheel(hrefs []string) string {
	var best string
	bestPriority := -1
	for _, href := range hrefs {
		decoded, err := url.QueryUnescape(href)
		if err != nil {
			decoded = href
		}
		// The PJRT package is "jax_rocm*_pjrt"; the compiler plugin is
		// "jax_rocm*_plugin". We only want the PJRT one.
		if !strings.Contains(decoded, "pjrt") || !strings.HasSuffix(decoded, ".whl") {
			continue
		}
		if p := rocmWheelPriority(decoded); p > bestPriority {
			best, bestPriority = href, p
		}
	}
	return best
}

// rocmWheelPriority ranks a ROCm wheel filename by platform compatibility.
func rocmWheelPriority(name string) int {
	switch {
	case strings.Contains(name, "manylinux_2_28"):
		return 3
	case strings.Contains(name, "manylinux2014"):
		return 2
	case strings.Contains(name, "manylinux"):
		return 1
	default:
		return 0
	}
}

// hrefRegex extracts the `href="..."` attributes from an HTML directory listing.
var hrefRegex = regexp.MustCompile(`href="([^"]+)"`)

// fetchHTMLHrefs fetches the given URL and returns the list of href targets it
// contains (used to parse simple HTML directory listings).
func fetchHTMLHrefs(url string) ([]string, error) {
	resp, err := http.Get(url)
	if err != nil {
		return nil, errors.Wrapf(err, "failed to fetch %s", url)
	}
	defer func() { ReportError(resp.Body.Close()) }()

	if resp.StatusCode != http.StatusOK {
		return nil, errors.Errorf("unexpected status code %d fetching %s", resp.StatusCode, url)
	}
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, errors.Wrap(err, "failed to read response body")
	}
	matches := hrefRegex.FindAllStringSubmatch(string(body), -1)
	hrefs := make([]string, 0, len(matches))
	for _, m := range matches {
		hrefs = append(hrefs, m[1])
	}
	return hrefs, nil
}
