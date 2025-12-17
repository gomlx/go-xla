package installer

import (
	"archive/tar"
	"archive/zip"
	"compress/gzip"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"

	"github.com/pkg/errors"
	"k8s.io/klog/v2"
)

const BinaryCPUReleasesRepo = "gomlx/pjrt-cpu-binaries"

// DownloadURLToTemp downloads a file from a given URL to a temporary file.
//
// It displays a spinner while downloading and outputs some information about the download.
//
// If useCache is true, it will save the file in a cache directory and try to reuse it if already downloaded.
//
// If wantSHA256 is not empty, it will verify the hash of the downloaded file.
//
// It returns the path where the file was downloaded, and if the downloaded file is in a cache
// (so it shouldn't be removed after use).
func DownloadURLToTemp(url, fileName, wantSHA256 string, useCache bool, verbosity VerbosityLevel) (
	filePath string, cached bool, err error) {
	// Download the asset to a temporary file
	var downloadedFile *os.File
	var renameTo string
	if useCache {
		filePath, cached, err = GetCachePath(fileName)
		if err != nil {
			return "", false, err
		}
		if !cached {
			renameTo = filePath
			filePath = filePath + ".tmp" // Download to temporary file first.
			downloadedFile, err = os.Create(filePath)
			if err != nil {
				return "", false, errors.Wrapf(err, "failed to create cache file %s", filePath)
			}
		}

	} else {
		// Create a temporary file.
		filePattern := fileName + ".*.tar.gz"
		downloadedFile, err = os.CreateTemp("", filePattern)
		if err != nil {
			return "", false, errors.Wrap(err, "failed to create temporary file")
		}
		filePath = downloadedFile.Name()
	}

	var downloadedBytesStr string
	if !cached {
		// Actually download the file.
		var bytesDownloaded int64
		spinnerErr := NewSpinner().
			Title(fmt.Sprintf("Downloading %s….", url)).
			Action(func(titleChange chan<- string) {
				var resp *http.Response
				resp, err = http.Get(url)
				if err != nil {
					err = errors.Wrapf(err, "failed to download asset %s", url)
					return
				}
				defer func() { ReportError(resp.Body.Close()) }()

				// Copy 1MB at a time, update the title with current bytes downloaded
				const bufSize = 1024 * 1024 // 1MB
				buffer := make([]byte, bufSize)
				bytesDownloaded = 0
				for {
					n, readErr := resp.Body.Read(buffer)
					if n > 0 {
						written, writeErr := downloadedFile.Write(buffer[:n])
						if writeErr != nil {
							err = errors.Wrapf(writeErr, "failed to write to file %s", downloadedFile.Name())
							break
						}
						bytesDownloaded += int64(written)
						// Update spinner title
						titleChange <- fmt.Sprintf("Downloading %s (%s) ...", url, formatBytes(bytesDownloaded))
					}
					if readErr == io.EOF {
						break
					}
					if readErr != nil {
						err = errors.Wrapf(readErr, "failed to download asset %s", url)
						break
					}
				}
				if err != nil {
					err = errors.Wrapf(err, "failed to write asset %s to temporary file %s", url, downloadedFile.Name())
					return
				}
				ReportError(downloadedFile.Close())
			}).
			Run()
		if spinnerErr != nil {
			return "", false, errors.Wrapf(spinnerErr, "failed run spinner for download from %s", url)
		}
		if err != nil {
			return "", false, err
		}
		downloadedBytesStr = formatBytes(bytesDownloaded)
	}

	// Verify SHA256 hash if provided -- also for cached files.
	verifiedStatus := ""
	if wantSHA256 != "" {
		// Open the file for reading.
		f, err := os.Open(filePath)
		if err != nil {
			return "", false, errors.Wrap(err, "failed to open file for hash verification")
		}
		defer func() { ReportError(f.Close()) }()

		// Calculate SHA256 hash using 1MB buffer
		hasher := sha256.New()
		buffer := make([]byte, 1024*1024) // 1MB buffer
		for {
			n, err := f.Read(buffer)
			if n > 0 {
				hasher.Write(buffer[:n])
			}
			if err == io.EOF {
				break
			}
			if err != nil {
				return "", false, errors.Wrap(err, "failed to read file for hash verification")
			}
		}

		actualHash := hex.EncodeToString(hasher.Sum(nil))
		if actualHash != wantSHA256 {
			return "", false, errors.Errorf("SHA256 hash mismatch for %s: expected %q, got %q", filePath, wantSHA256, actualHash)
		}
		verifiedStatus = " (hash checked)"
	}

	// If downloaded to a temporary file, rename to final destination:
	if renameTo != "" {
		_ = os.Remove(renameTo)
		if err := os.Rename(filePath, renameTo); err != nil {
			return "", false, errors.Wrapf(err, "failed to rename %s to %s", filePath, renameTo)
		}
		filePath = renameTo
		renameTo = ""
	}

	if cached {
		switch verbosity {
		case Verbose:
			fmt.Printf("- Reusing %s from cache%s\n", filePath, verifiedStatus)
		case Normal:
			fmt.Printf("\r- Reusing %s from cache%s%s", filePath, verifiedStatus, DeleteToEndOfLine)
		case Quiet:
		}
	} else {
		switch verbosity {
		case Verbose:
			fmt.Printf("- Downloaded %s to %s%s\n", downloadedBytesStr, filePath, verifiedStatus)
		case Normal:
			fmt.Printf("\r- Downloaded %s to %s%s%s", downloadedBytesStr, filePath, verifiedStatus, DeleteToEndOfLine)
		case Quiet:
		}
		if useCache {
			// Now the file is cached.
			cached = true
		}
	}
	return filePath, cached, nil
}

// ExtractFileFromZip searches for a file named fileName within the zipFilePath
// and extracts the first one found to the outputPath.
//
// The fileName is matched to the full path of the files in the zip archive as well as
// to the base name: so a base name can be given, and it will find it anywhere in the zip archive.
func ExtractFileFromZip(zipFilePath, fileName, outputPath string) error {
	r, err := zip.OpenReader(zipFilePath)
	if err != nil {
		return err
	}
	defer func() { ReportError(r.Close()) }()

	// Normalize the target file name for comparison
	normalizedTarget := filepath.Clean(fileName)

	// Iterate through the files in the archive
	for _, f := range r.File {
		// Identical match:
		if f.Name == normalizedTarget {
			return extractZipFile(f, outputPath)
		}

		// Base name match:
		_, baseName := filepath.Split(f.Name)
		if baseName == normalizedTarget {
			return extractZipFile(f, outputPath)
		}
	}
	return os.ErrNotExist // File was not found in the archive.
}

// ExtractDirFromZip extracts from zipFilePath all files and directories under dirInZipFile and saves them with the
// same directory structure under outputPath.
//
// Notice dirInZipFile is not repeated in outputPath.
func ExtractDirFromZip(zipFilePath, dirInZipFile, outputPath string) error {
	r, err := zip.OpenReader(zipFilePath)
	if err != nil {
		return err
	}
	defer func() { ReportError(r.Close()) }()

	// Normalize paths for comparison
	normalizedPrefix := filepath.Clean(dirInZipFile) + "/"

	// Iterate through the files in the archive
	for _, f := range r.File {
		// Normalize the file path
		normalizedPath := filepath.Clean(f.Name)

		// Check if this file is under the requested directory
		if !strings.HasPrefix(normalizedPath, normalizedPrefix) {
			continue
		}

		// Calculate relative path from the prefix
		relPath := strings.TrimPrefix(normalizedPath, normalizedPrefix)
		if relPath == "" {
			continue // Skip the directory itself
		}

		// Create the full output path
		fullPath := filepath.Join(outputPath, relPath)

		if f.FileInfo().IsDir() {
			// Create directory
			if err := os.MkdirAll(fullPath, 0755); err != nil {
				return err
			}
			continue
		}

		// Create parent directories if they don't exist
		if err := os.MkdirAll(filepath.Dir(fullPath), 0755); err != nil {
			return err
		}

		if err := extractZipFile(f, fullPath); err != nil {
			return err
		}
	}
	return nil
}

// extractZipFile is a helper to perform the actual extraction
//
// It adds execution permissions to the file if it is in the bin directory.
func extractZipFile(f *zip.File, outputPath string) error {
	rc, err := f.Open()
	if err != nil {
		return err
	}
	defer func() { ReportError(rc.Close()) }()

	// Use the file mode from the zip entry
	fMode := f.Mode()
	if fMode == 0 {
		// Fallback if zip doesn't have mode set
		fMode = 0644
	}

	// Create the output file
	outFile, err := os.OpenFile(outputPath, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, fMode)
	if err != nil {
		return err
	}
	defer func() { ReportError(outFile.Close()) }()

	// Copy the contents
	_, err = io.Copy(outFile, rc)
	if err != nil {
		return err
	}

	// Make file executable if in a bin directory
	if strings.Contains(outputPath, "/bin/") {
		if err := os.Chmod(outputPath, 0755); err != nil {
			return err
		}
	}
	return nil
}

// GitHubGetLatestVersion returns the latest version tag from the gomlx/pjrt-cpu-binaries repository.
func GitHubGetLatestVersion() (string, error) {
	latestURL := fmt.Sprintf("https://api.github.com/repos/%s/releases/latest", BinaryCPUReleasesRepo)
	retries := 0
	const maxRetries = 2
retry:
	for {
		// Make HTTP request with optional authorization header
		req, err := http.NewRequest("GET", latestURL, nil)
		if err != nil {
			return "", errors.Wrapf(err, "failed to create request for %q", latestURL)
		}
		req.Header.Add("Accept", "application/vnd.github+json")
		if token, found := os.LookupEnv("GH_TOKEN"); found {
			if token == "" {
				klog.V(1).Infof("GH_TOKEN is empty, skipping authentication")
			} else {
				req.Header.Add("Authorization", "Bearer "+token)
				klog.V(1).Infof("Using GitHub token for authentication")
			}
		} else {
			klog.V(1).Infof("GH_TOKEN is not set, skipping authentication")
		}
		resp, err := http.DefaultClient.Do(req)
		if err != nil {
			return "", errors.Wrapf(err, "failed to fetch release data from %q", latestURL)
		}

		// Read response body
		body, err := io.ReadAll(resp.Body)
		ReportError(resp.Body.Close())
		if err != nil {
			return "", errors.Wrapf(err, "failed to read data from %q", latestURL)
		}
		if resp.StatusCode != http.StatusOK {
			return "", errors.Errorf("failed to get version from %q, got status code %d -- message %q", latestURL, resp.StatusCode, body)
		}

		// Parse JSON response
		var info struct {
			TagName string `json:"tag_name"`
		}
		if err := json.Unmarshal(body, &info); err != nil {
			return "", errors.Wrapf(err, "failed to parse JSON response")
		}
		version := info.TagName
		if version == "" {
			if retries == maxRetries {
				return "", errors.Errorf("failed to get version from %q, it is missing the field `tag_name`", latestURL)
			}
			retries++
			klog.Warningf("failed to get version from %q, it is missing the field `tag_name`, retrying...", latestURL)
			fmt.Printf("Body: %s\n", string(body))
			continue retry
		}
		return version, nil
	}
}

// GitHubDownloadReleaseAssets downloads the list of assets available for the given repository/release version.
// E.g.: repo = "gomlx/pjrt-cpu-binaries", version = "v0.83.1"
func GitHubDownloadReleaseAssets(repo string, version string) ([]string, error) {
	// Construct release URL based on the version -- "latest" is not supported at this point.
	releaseURL := fmt.Sprintf("https://api.github.com/repos/%s/releases/tags/%s", repo, version)

	// Make HTTP request with optional authorization header
	req, err := http.NewRequest("GET", releaseURL, nil)
	if err != nil {
		return nil, errors.Wrapf(err, "failed to create request for %q", releaseURL)
	}
	req.Header.Add("Accept", "application/vnd.github+json")
	if token := os.Getenv("GH_TOKEN"); token != "" {
		req.Header.Add("Authorization", "Bearer "+token)
		klog.V(1).Infof("Using GitHub token for authentication")
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return nil, errors.Wrapf(err, "failed to fetch release data from %q", releaseURL)
	}

	// Check response status code
	if resp.StatusCode == http.StatusNotFound {
		return nil, errors.Errorf("version %q not found", version)
	}
	if resp.StatusCode != http.StatusOK {
		if resp.StatusCode == http.StatusForbidden {
			return nil, errors.Errorf(
				"unexpected status code %d - %q (url=%q); maybe it's beeing throttled by GitHub, "+
					"and requires GH_TOKEN to be set?",
				resp.StatusCode, resp.Status, releaseURL)
		}
		return nil, errors.Errorf("unexpected status code %d - %q (url=%q)", resp.StatusCode, resp.Status, releaseURL)
	}

	// Read response body
	body, err := io.ReadAll(resp.Body)
	ReportError(resp.Body.Close())
	if err != nil {
		return nil, errors.Wrapf(err, "failed to read response body")
	}

	// Parse JSON response
	var release struct {
		Assets []struct {
			BrowserDownloadURL string `json:"browser_download_url"`
		} `json:"assets"`
	}
	if err := json.Unmarshal(body, &release); err != nil {
		return nil, errors.Wrapf(err, "failed to parse JSON response")
	}

	// Extract .tar.gz download URLs
	var urls []string
	for _, asset := range release.Assets {
		if strings.HasSuffix(asset.BrowserDownloadURL, ".tar.gz") {
			urls = append(urls, asset.BrowserDownloadURL)
		}
	}

	return urls, nil
}

func GitHubGetVersions(repo string) ([]string, error) {
	resp, err := http.Get(fmt.Sprintf("https://api.github.com/repos/%s/releases", repo))
	if err != nil {
		return nil, errors.Wrapf(err, "failed to fetch versions from GitHub")
	}
	defer func() { ReportError(resp.Body.Close()) }()

	if resp.StatusCode != http.StatusOK {
		return nil, errors.Errorf("unexpected status code %d", resp.StatusCode)
	}

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, errors.Wrapf(err, "failed to read response body")
	}

	// Parse JSON response
	var releases []struct {
		Name string `json:"name"`
		Tag  string `json:"tag_name"`
	}
	if err := json.Unmarshal(body, &releases); err != nil {
		return nil, errors.Wrapf(err, "failed to parse JSON response")
	}

	// Extract version names
	var versions []string
	for _, release := range releases {
		versions = append(versions, release.Tag)
	}

	return versions, nil
}

// Untar takes a path to a tar/gzip file and an output directory.
// It returns a list of extracted files and any error encountered.
func Untar(tarballPath, outputDirPath string) ([]string, error) {
	// Make sure the output directory is absolute.
	if !filepath.IsAbs(outputDirPath) {
		var err error
		outputDirPath, err = filepath.Abs(outputDirPath)
		if err != nil {
			return nil, errors.Wrapf(err, "Untar failed to get absolute path for output directory %q", outputDirPath)
		}
	}

	// 1. Open the tarball file
	file, err := os.Open(tarballPath)
	if err != nil {
		return nil, errors.Wrapf(err, "failed to open tarball in %s for reading", tarballPath)
	}
	defer func() { ReportError(file.Close()) }()

	// 2. Setup the Gzip reader (assuming it's a .tar.gz)
	// If it's just a .tar file, you would skip this step and use 'file' directly below.
	var fileReader io.Reader = file
	if filepath.Ext(tarballPath) == ".gz" || filepath.Ext(tarballPath) == ".tgz" {
		gzReader, err := gzip.NewReader(file)
		if err != nil {
			return nil, errors.Wrapf(err, "failed to create gzip reader")
		}
		defer func() { ReportError(gzReader.Close()) }()
		fileReader = gzReader
	}

	// 3. Setup the Tar reader
	tarReader := tar.NewReader(fileReader)

	// Track extracted files
	var extractedFiles []string

	// 4. Iterate through the archive entries
	for {
		header, err := tarReader.Next()
		if err == io.EOF {
			break // End of archive
		}
		if err != nil {
			return nil, errors.Wrapf(err, "tar reading error")
		}

		// Clean the targetPath and make sure it falls within outputDirPath.
		targetPath := filepath.Join(outputDirPath, header.Name)
		targetPath = filepath.Clean(targetPath)
		if !strings.HasPrefix(targetPath, outputDirPath) {
			return nil, errors.Errorf("tar entry path is unsafe: %s", header.Name)
		}

		// Create parent directories if they don't exist
		if err := os.MkdirAll(filepath.Dir(targetPath), 0755); err != nil {
			return nil, errors.Wrapf(err, "failed to create directory %s", filepath.Dir(targetPath))
		}

		switch header.Typeflag {
		case tar.TypeDir:
			// Handle directories
			if err := os.MkdirAll(targetPath, os.FileMode(header.Mode)); err != nil {
				return nil, errors.Wrapf(err, "failed to create directory %s", targetPath)
			}
			extractedFiles = append(extractedFiles, targetPath)

		case tar.TypeReg:
			// Handle regular files and links
			outFile, err := os.OpenFile(targetPath, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, os.FileMode(header.Mode))
			if err != nil {
				// If file exists and is read-only, remove it and try again.
				if os.IsPermission(err) {
					if err = os.Remove(targetPath); err != nil {
						return nil, errors.Wrapf(err, "failed to remove read-only file %s", targetPath)
					}
					outFile, err = os.OpenFile(targetPath, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, os.FileMode(header.Mode))
					if err != nil {
						return nil, errors.Wrapf(err, "failed to create file %s after removing read-only version", targetPath)
					}
				} else {
					return nil, errors.Wrapf(err, "failed to create file %s", targetPath)
				}
			}
			// Copy file contents
			if _, err := io.Copy(outFile, tarReader); err != nil {
				ReportError(outFile.Close())
				return nil, errors.Wrapf(err, "failed to copy file contents to %s", targetPath)
			}
			ReportError(outFile.Close())
			extractedFiles = append(extractedFiles, targetPath)

		case tar.TypeSymlink:

			// Sanitize the symlink's target path to ensure it stays within the output directory
			linkTarget := filepath.Clean(header.Linkname)
			if filepath.IsAbs(linkTarget) {
				return nil, errors.Errorf("absolute symlink target path unsafe and not allowed: %s", linkTarget)
			}
			// Calculate the absolute path of the link target relative to the symlink's location
			absLinkTarget := filepath.Join(filepath.Dir(targetPath), linkTarget)
			cleanAbsTarget, err := filepath.EvalSymlinks(absLinkTarget)
			if err != nil && !os.IsNotExist(err) {
				return nil, errors.Wrapf(err, "failed to evaluate symlink target for %s", targetPath)
			}
			if cleanAbsTarget != "" && !strings.HasPrefix(cleanAbsTarget, outputDirPath) {
				return nil, errors.Errorf("symlink target path escapes output directory: %s", linkTarget)
			}
			// Remove the target file if it exists.
			if err := os.Remove(targetPath); err != nil && !os.IsNotExist(err) {
				return nil, errors.Wrapf(err, "failed to remove existing file at symlink target %s", targetPath)
			}
			// Create the symlink
			if err := os.Symlink(linkTarget, targetPath); err != nil {
				return nil, errors.Wrapf(err, "failed to create symlink %s -> %s", targetPath, linkTarget)
			}
			extractedFiles = append(extractedFiles, targetPath)

		default:
			klog.Errorf("Skipping unsupported type: %c for file %s\n", header.Typeflag, header.Name)
		}
	}
	return extractedFiles, nil
}

// Unzip takes a path to a zip file and an output directory.
// It returns a list of extracted files and any error encountered.
func Unzip(zipPath, outputDirPath string) ([]string, error) {
	// Make sure the output directory is absolute.
	if !filepath.IsAbs(outputDirPath) {
		var err error
		outputDirPath, err = filepath.Abs(outputDirPath)
		if err != nil {
			return nil, errors.Wrapf(err, "Unzip failed to get absolute path for output directory %q", outputDirPath)
		}
	}

	r, err := zip.OpenReader(zipPath)
	if err != nil {
		return nil, errors.Wrapf(err, "failed to open zip file %s", zipPath)
	}
	defer func() { ReportError(r.Close()) }()

	var extractedFiles []string

	for _, f := range r.File {
		// Clean the targetPath and make sure it falls within outputDirPath.
		targetPath := filepath.Join(outputDirPath, f.Name)
		targetPath = filepath.Clean(targetPath)
		if !strings.HasPrefix(targetPath, outputDirPath) {
			return nil, errors.Errorf("zip entry path is unsafe: %s", f.Name)
		}

		if f.FileInfo().IsDir() {
			// Create directory
			if err := os.MkdirAll(targetPath, f.Mode()); err != nil {
				return nil, errors.Wrapf(err, "failed to create directory %s", targetPath)
			}
			extractedFiles = append(extractedFiles, targetPath)
			continue
		}

		// Create parent directories if they don't exist
		if err := os.MkdirAll(filepath.Dir(targetPath), 0755); err != nil {
			return nil, errors.Wrapf(err, "failed to create directory %s", filepath.Dir(targetPath))
		}

		// Handle regular files using the helper
		if err := extractZipFile(f, targetPath); err != nil {
			return nil, errors.Wrapf(err, "failed to extract file %s", targetPath)
		}
		extractedFiles = append(extractedFiles, targetPath)
	}

	return extractedFiles, nil
}
