package installer

import (
	"fmt"
	"os"
	"os/user"
	"path/filepath"
	"runtime"
	"strings"
	"unicode/utf8"

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

	// Accept either '/' or '\' as separator following the user name.
	sepIdx := -1
	if runtime.GOOS != "windows" {
		sepIdx = strings.IndexRune(dir, filepath.Separator)
	} else {
		// In windows we accept both: "/" and "\\".
		sepIdxUnix := strings.IndexRune(dir, '/')
		sepIdxWin := strings.IndexRune(dir, '\\')

		// Find the earliest separator (if any)
		if sepIdxUnix == -1 {
			sepIdx = sepIdxWin
		} else if sepIdxWin == -1 {
			sepIdx = sepIdxUnix
		} else if sepIdxUnix < sepIdxWin {
			sepIdx = sepIdxUnix
		} else {
			sepIdx = sepIdxWin
		}
	}

	// Find user name after the tilde, if one is given.
	var userName string
	if dir != "~" && sepIdx != 1 { // "~/" or "~\\"
		// Extract the username, whatever the first separator is
		if sepIdx == -1 {
			userName = dir[1:]
		} else {
			userName = dir[1:sepIdx]
		}
	}

	// Retrive user and their home directory.
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
	// Replace ~ or ~user with user home, preserve any following path, no matter the separator
	remaining := ""
	if userName == "" {
		remaining = dir[1:]
	} else {
		remaining = dir[1+len(userName):]
	}
	// If remaining starts with '/' or '\', remove it so Join works as expected.
	remaining = strings.TrimLeft(remaining, `/\`)

	return filepath.Join(homeDir, remaining), nil
}

// formatBytes formats bytes into a human-readable string (e.g., 1.5 MB)
func formatBytes(b int64) string {
	const unit = 1024
	if b < unit {
		return fmt.Sprintf("%d B", b)
	}
	div, exp := int64(unit), 0
	for ; b >= div*unit && exp < 5; div *= unit {
		exp++
	}
	return fmt.Sprintf("%.1f %cB", float64(b)/float64(div), "KMGTPE"[exp])
}

// truncateToWidth truncates a string to a given width, adding ellipsis in the middle if necessary.
func truncateToWidth(s string, width int) string {
	sLen := utf8.RuneCountInString(s)
	if sLen < width {
		return s
	}

	// Convert to runes to slice safely (avoid cutting multi-byte chars)
	r := []rune(s)

	// Extreme cases.
	if width <= 3 {
		return string(r[:width])
	}

	// Truncate in the middle and add ellipsis, ensuring we fit exactly
	copy(r[width/2+2:], r[sLen-(width/2-2):])
	r[width/2-1] = '.'
	r[width/2] = '.'
	r[width/2+1] = '.'
	return string(r[:width-3]) + "…"
}
