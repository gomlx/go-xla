package installer

import (
	"os"
	"os/user"
	"path"
	"path/filepath"
	"strings"

	"github.com/pkg/errors"
	"k8s.io/klog/v2"
)

type VerbosityLevel int

const (
	Quiet VerbosityLevel = iota
	Normal
	Verbose
)

const DeleteToEndOfLine = "\x1b[J"

// ReportError prints an error if it is not nil, but otherwise does nothing.
func ReportError(err error) {
	if err != nil {
		klog.Warningf("Error: %v", err)
	}
}

// GetCachePath finds and prepares the cache directory for gopjrt.
//
// It uses os.UserCacheDir() for portability:
//
// - Linux: $XDG_CACHE_HOME or $HOME/.cache
// - Darwin: $HOME/Library/Caches
// - Windows: %LocalAppData% (e.g., C:\Users\user\AppData\Local)
func GetCachePath(fileName string) (filePath string, cached bool, err error) {
	baseCacheDir, err := os.UserCacheDir()
	if err != nil {
		return "", false, errors.Wrap(err, "failed to find user cache directory")
	}
	cacheDir := filepath.Join(baseCacheDir, "go-xla")
	if err = os.MkdirAll(cacheDir, 0755); err != nil {
		return "", false, errors.Wrapf(err, "failed to create cache directory %s", cacheDir)
	}
	filePath = filepath.Join(cacheDir, fileName)
	if stat, err := os.Stat(filePath); err == nil {
		cached = stat.Mode().IsRegular()
	}
	return
}

// ReplaceTildeInDir replaces "~" in a directory path with the user's home directory.
// Returns dir if it doesn't start with "~".
// It may panic with an error if `dir` has an unknown user (e.g: `~unknown/...`)
func ReplaceTildeInDir(dir string) (string, error) {
	if len(dir) == 0 {
		return "", nil
	}
	if dir[0] != '~' {
		return dir, nil
	}
	var userName string
	if dir != "~" && !strings.HasPrefix(dir, "~/") {
		sepIdx := strings.IndexRune(dir, '/')
		if sepIdx == -1 {
			userName = dir[1:]
		} else {
			userName = dir[1:sepIdx]
		}
	}
	var usr *user.User
	var err error
	if userName == "" {
		usr, err = user.Current()
	} else {
		usr, err = user.Lookup(userName)
	}
	if err != nil {
		return "", errors.Wrapf(err, "failed to lookup home directory for user in path %q", dir)
	}
	homeDir := usr.HomeDir
	return path.Join(homeDir, dir[1+len(userName):]), nil
}
