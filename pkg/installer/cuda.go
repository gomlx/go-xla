//go:build (linux && amd64) || pjrt_all

package installer

import (
	"fmt"
	"maps"
	"os"
	"os/exec"
	"path"
	"path/filepath"
	"runtime"
	"slices"
	"strings"
	"sync"

	"github.com/pkg/errors"
	"k8s.io/klog/v2"
)

// HasNvidiaGPU tries to guess if there is an actual Nvidia GPU installed (as opposed to only the drivers/PJRT
// file installed, but no actual hardware).
// It does that by checking for the presence of the device files in /dev/nvidia*.
var HasNvidiaGPU = sync.OnceValue[bool](func() bool {
	matches, err := filepath.Glob("/dev/nvidia*")
	if err != nil {
		klog.Errorf("Failed to figure out if there is an Nvidia GPU installed while searching for files matching \"/dev/nvidia*\": %v", err)
	} else if len(matches) > 0 {
		return true
	}

	// Execute the nvidia-smi command if present
	_, lookErr := exec.LookPath("nvidia-smi")
	if lookErr != nil {
		return false
	}
	cmd := exec.Command("nvidia-smi")
	output, cmdErr := cmd.CombinedOutput()
	if cmdErr != nil {
		return false
	}
	return strings.Contains(string(output), "NVIDIA-SMI")
})

func init() {
	autoInstallers["cuda"] = CudaAutoInstall
}

// CudaAutoInstall installs the latest version of the CUDA PJRT and Nvidia libraries
// if not yet installed, and there is an actual Nvidia GPU installed.
//
// It uses HasNvidiaGPU to see if there is an actual Nvidia GPU installed.
func CudaAutoInstall(installPath string, useCache bool, verbosity VerbosityLevel) error {
	if runtime.GOOS != "linux" || runtime.GOARCH != "amd64" {
		// Only supported on Linux/amd64.
		return nil
	}
	pjrtPluginPath := path.Join(installPath, "nvidia", "pjrt_c_api_cuda_plugin.so")
	_, err := os.Stat(pjrtPluginPath)
	if err == nil {
		// Already installed.
		return nil
	}
	if !HasNvidiaGPU() {
		// No need to install anything.
		return nil
	}
	return CudaInstall("cuda13", "latest", installPath, useCache, verbosity)
}

// CudaInstall installs the cuda PJRT from the Jax PIP packages, using pypi.org distributed files.
//
// Checks performed:
// - Version exists
// - Author email is from the Jax team
// - Downloaded files sha256 match the ones on pypi.org
//
// # If useCache is true, it will save the file in a cache directory and try to reuse it if already downloaded
//
// The installPath parameter should be to the .../lib/go-xla directory. The pjrt plugin itself will be installed
// under the .../lib/go-xla/nvidia directory -- it needs to be there due to path resolution issues with the
// plugin/Nvidia libraries.
func CudaInstall(plugin, version, installPath string, useCache bool, verbosity VerbosityLevel) error {
	// Create the target directory.
	var err error
	installPath, err = ReplaceTildeInDir(installPath)
	if err != nil {
		return err
	}
	if err := os.MkdirAll(installPath, 0755); err != nil {
		return errors.Wrapf(err, "failed to create install directory in %s", installPath)
	}

	// Remove any previous version of the nvidia libraries and recreate it.
	nvidiaSubdir := filepath.Join(installPath, "nvidia")
	if err := os.RemoveAll(nvidiaSubdir); err != nil {
		return errors.Wrapf(err, "failed to remove existing nvidia libraries directory %s", nvidiaSubdir)
	}

	// Install PJRT plugin.
	version, err = CudaInstallPJRT(plugin, version, nvidiaSubdir, useCache, verbosity)
	if err != nil {
		return err
	}

	// Pinstall required Nvidia libraries.
	if err := CudaInstallNvidiaLibraries(plugin, version, nvidiaSubdir, useCache, verbosity); err != nil {
		return err
	}
	cudaVersion := "13"
	if plugin == "cuda12" {
		cudaVersion = "12"
	}
	if verbosity == Verbose {
		fmt.Println()
	}
	fmt.Printf("\r✅ Installed \"cuda\" PJRT and Nvidia libraries based on Jax version %s and CUDA version %s\n", version, cudaVersion)
	if verbosity == Verbose {
		fmt.Println()
	}
	return nil
}

// CudaInstallPJRT installs the cuda PJRT from the Jax PIP packages, using pypi.org distributed files.
//
// Checks performed:
// - Version exists
// - Author email is from the Jax team
// - Downloaded files sha256 match the ones on pypi.org
//
// If useCache is true, it will save the file in a cache directory and try to reuse it if already downloaded.
//
// Returns the version that was installed -- it can be different if the requested version was "latest", in which case it
// is translated to the actual version.
func CudaInstallPJRT(plugin, version, installPath string, useCache bool, verbosity VerbosityLevel) (string, error) {
	// Make the directory that will hold the PJRT files.
	if err := os.MkdirAll(installPath, 0755); err != nil {
		return "", errors.Wrapf(err, "failed to create PJRT install directory in %s", installPath)
	}
	pjrtOutputPath := path.Join(installPath, "pjrt_c_api_cuda_plugin.so")

	// Get CUDA PJRT wheel from pypi.org
	info, packageName, err := CudaGetPJRTPipInfo(plugin)
	if err != nil {
		return "", errors.WithMessagef(err, "can't fetch pypi.org information for %s", plugin)
	}
	if info.Info.AuthorEmail != "jax-dev@google.com" {
		return "", errors.Errorf("package %s is not from Jax team, but it's signed by %q: something is suspicious!?",
			packageName, info.Info.AuthorEmail)
	}

	// Translate "latest" to the actual version if needed.
	if version == "latest" {
		version = info.Info.Version
	}

	releaseInfos, ok := info.Releases[version]
	if !ok {
		versions := slices.Collect(maps.Keys(info.Releases))
		slices.Sort(versions)
		return "", errors.Errorf("version %q not found for %q (from pip package %q) -- lastest is %q and existing versions are: %s",
			version, plugin, packageName, info.Info.Version, strings.Join(versions, ", "))
	}

	releaseInfo, err := PipSelectRelease(releaseInfos, PipPackageLinuxAMD64(), false)
	if err != nil {
		return "", errors.Wrapf(err, "failed to find release for %s, version %s", plugin, version)
	}
	if releaseInfo.PackageType != "bdist_wheel" {
		return "", errors.Errorf("release %s is not a \"binary wheel\" type", releaseInfo.Filename)
	}

	sha256hash := releaseInfo.Digests["sha256"]
	downloadedJaxPJRTWHL, fileCached, err := DownloadURLToTemp(releaseInfo.URL, fmt.Sprintf("go-xla_%s_%s.whl", packageName, version), sha256hash, useCache, verbosity)
	if err != nil {
		return "", errors.Wrap(err, "failed to download cuda PJRT wheel")
	}
	if !fileCached {
		defer func() { ReportError(os.Remove(downloadedJaxPJRTWHL)) }()
	}
	err = ExtractFileFromZip(downloadedJaxPJRTWHL, "xla_cuda_plugin.so", pjrtOutputPath)
	if err != nil {
		return "", errors.Wrapf(err, "failed to extract CUDA PJRT file from %q wheel", packageName)
	}
	switch verbosity {
	case Verbose:
		fmt.Printf("- Installed %s %s to %s\n", plugin, version, pjrtOutputPath)
	case Normal:
		fmt.Printf("\r- Installed %s %s to %s%s", plugin, version, pjrtOutputPath, DeleteToEndOfLine)
	case Quiet:
	}
	return version, nil
}

// CudaValidateVersion checks whether the cuda version selected by "-version" exists.
func CudaValidateVersion(plugin, version string) error {
	// "latest" is always valid.
	if version == "latest" {
		return nil
	}

	info, packageName, err := CudaGetPJRTPipInfo(plugin)
	if err != nil {
		return errors.WithMessagef(err, "can't fetch pypi.org information for %s", plugin)
	}
	if info.Info.AuthorEmail != "jax-dev@google.com" {
		return errors.Errorf("package %s is not from Jax team, but it's signed by %q: something is suspicious!?",
			packageName, info.Info.AuthorEmail)
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

// CudaGetPJRTPipInfo returns the JSON info for the PIP package that corresponds to the plugin.
func CudaGetPJRTPipInfo(plugin string) (*PipPackageInfo, string, error) {
	var packageName string
	switch plugin {
	case "cuda12":
		packageName = "jax-cuda12-pjrt"
	case "cuda13":
		packageName = "jax-cuda13-pjrt"
	default:
		return nil, "", errors.Errorf("unknown plugin %q selected", plugin)
	}
	info, err := GetPipInfo(packageName)
	if err != nil {
		return nil, "", errors.Wrapf(err, "failed to get package info for %s", packageName)
	}
	return info, packageName, nil
}

// CudaInstallNvidiaLibraries installs the required NVIDIA libraries for CUDA.
func CudaInstallNvidiaLibraries(plugin, version, nvidiaSubdir string, useCache bool, verbosity VerbosityLevel) error {
	// Find required nvidia packages:
	packageName := "jax-" + plugin + "-plugin"
	jaxCudaPluginInfo, err := GetPipInfo(packageName)
	if err != nil {
		return errors.Wrapf(err, "failed to fetch the package info for %s", packageName)
	}
	if verbosity == Verbose {
		fmt.Println("Dependencies:")
	}
	deps, err := jaxCudaPluginInfo.ParseDependencies()
	if err != nil {
		return errors.Wrapf(err, "failed to parse the dependencies for %s", packageName)
	}
	nvidiaDependencies := slices.DeleteFunc(deps, func(dep PipDependency) bool {
		// This is a simplification that works for now: in the future we many need to check "sys_platform" conditions.
		if !strings.HasPrefix(dep.Package, "nvidia") {
			return true
		}
		return false
	})

	// Install the nvidia libraries found in the dependencies.
	for _, dep := range nvidiaDependencies {
		err = cudaInstallNvidiaLibrary(nvidiaSubdir, dep, useCache, verbosity)
		if err != nil {
			return err
		}
	}

	// Create a link to the binary ptxas, required by the nvidia libraries.
	nvidiaBinPath := filepath.Join(nvidiaSubdir, "bin")
	if err := os.MkdirAll(nvidiaBinPath, 0755); err != nil {
		return errors.Wrapf(err, "failed to create nvidia bin directory in %s", nvidiaBinPath)
	}

	// Create symbolic link to ptxas.
	var ptxasPath string
	switch plugin {
	case "cuda12":
		ptxasPath = filepath.Join(nvidiaSubdir, "cuda_nvcc/bin/ptxas")
	case "cuda13":
		ptxasPath = filepath.Join(nvidiaSubdir, "cu13/bin/ptxas")
	default:
		return errors.Errorf("version validation not implemented for plugin %q in version %s", plugin, version)
	}
	ptxasLinkPath := filepath.Join(nvidiaBinPath, "ptxas")
	if err := os.Symlink(ptxasPath, ptxasLinkPath); err != nil {
		return errors.Wrapf(err, "failed to create symbolic link to ptxas in %s", ptxasLinkPath)
	}

	// Link libraries that Nvidia is not able to find from the SDK path set.
	libsPath := path.Dir(path.Dir(nvidiaSubdir))
	switch plugin {
	case "cuda13":
		// Source of the symlink relative to the `lib` path.
		libCublasPath := "./go-xla/nvidia/cu13/lib"
		for _, srcName := range []string{"libcublasLt.so.13", "libcublas.so.13"} {
			dstPath := filepath.Join(libsPath, filepath.Base(srcName))
			srcPath := filepath.Join(libCublasPath, srcName)
			if err := os.Remove(dstPath); err != nil && !os.IsNotExist(err) {
				return errors.Wrapf(err, "failed to remove existing symlink to %s in %s", srcPath, dstPath)
			}
			if err := os.Symlink(srcPath, dstPath); err != nil {
				return errors.Wrapf(err, "failed to create symbolic link to %s in %s", srcPath, dstPath)
			}
		}
	}
	return nil
}

func cudaInstallNvidiaLibrary(nvidiaSubdir string, dep PipDependency, useCache bool, verbosity VerbosityLevel) error {
	info, err := GetPipInfo(dep.Package)
	if err != nil {
		return errors.Wrapf(err, "failed to fetch the package info for %s", dep.Package)
	}

	// Find the highest version that meets constraints.
	var selectedVersion string
	var selectedReleaseInfo *PipReleaseInfo
	for version, releases := range info.Releases {
		if !dep.IsValid(version) {
			continue
		}
		releaseInfo, err := PipSelectRelease(releases, PipPackageLinuxAMD64(), false)
		if err != nil {
			continue
		}
		if selectedVersion == "" || PipCompareVersion(version, selectedVersion) > 0 {
			selectedVersion = version
			selectedReleaseInfo = releaseInfo
		}
	}
	if selectedVersion == "" {
		return errors.Errorf("no matching version found for package %s with constraints %+v", dep.Package, dep)
	}

	// Download the ".whl" file (zip file format) for the selected version of the nvidia library..
	sha256hash := selectedReleaseInfo.Digests["sha256"]
	downloadedWHL, whlIsCached, err := DownloadURLToTemp(selectedReleaseInfo.URL, fmt.Sprintf("go-xla_%s_%s.whl", dep.Package, selectedVersion), sha256hash, useCache, verbosity)
	if err != nil {
		return errors.Wrapf(err, "failed to download %s wheel", dep.Package)
	}
	if !whlIsCached {
		defer func() { ReportError(os.Remove(downloadedWHL)) }()
	}

	// Extract all files under "nvidia/" for this package.
	if err := ExtractDirFromZip(downloadedWHL, "nvidia", nvidiaSubdir); err != nil {
		return errors.Wrapf(err, "failed to extract nvidia libraries from %s", downloadedWHL)
	}
	switch verbosity {
	case Verbose:
		fmt.Printf("- Installed %s@%s\n", dep.Package, selectedVersion)
	case Normal:
		fmt.Printf("\r- Installed %s@%s%s", dep.Package, selectedVersion, DeleteToEndOfLine)
	}
	return nil
}
