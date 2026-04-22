package installer

import (
	"fmt"
	"os"
	"path/filepath"
	"time"

	"github.com/gofrs/flock"
	"github.com/pkg/errors"
)

// fileLock represents a locked file for the installation path.
// Make sure to always call Unlock() when you are done with the lock -- recommend using defer.
type fileLock struct {
	installFile, lockPath string
	flock                 *flock.Flock
}

var (
	// InstallationFileLockTimeout is the timeout for acquiring the file lock.
	// If it waits for more than those many minutes it returns a timeout error.
	InstallationFileLockTimeout = 5 * time.Minute

	// RetryLockPeriod is the period to wait between attempts to acquire the file lock.
	RetryLockPeriod = 1000 * time.Millisecond
)

// checkInstallOrFileLock checks if the plugin is installed, and if it is not, it creates a file lock
// on it, which is returned.
func checkInstallOrFileLock(installFile string) (isInstalled bool, fLock *fileLock, err error) {
	// Check whether the file is already installed.
	installFile, err = ReplaceTildeInDir(installFile)
	if err != nil {
		return false, nil, err
	}
	_, err = os.Stat(installFile)
	if err == nil {
		return true, nil, nil
	}
	if !os.IsNotExist(err) {
		// Disk error, permissions or something else ?
		return false, nil, errors.Wrapf(err, "failed to stat install file %q", installFile)
	}

	// Make sure the directory exists for the installation and lock files.
	if err := os.MkdirAll(filepath.Dir(installFile), 0755); err != nil {
		return false, nil, errors.Wrap(err, "failed to create install directory %q")
	}

	// Try to acquire the lock.
	lockPath := fmt.Sprintf("%s.lock", installFile)
	fLock = &fileLock{installFile: installFile, lockPath: lockPath}
	fLock.flock = flock.New(lockPath)
	timeOut := time.After(InstallationFileLockTimeout)
	var ok bool
	for {
		ok, err = fLock.flock.TryLock()
		if err != nil {
			return false, nil, errors.Wrapf(err, "failed to acquire lock %q for install file %q", lockPath, installFile)
		}
		if ok {
			return false, fLock, nil
		}
		select {
		case <-timeOut:
			return false, nil, errors.Errorf(
				"timeout waiting for lock in %q: either there is a slow installation in progress, "+
					"or the lock file %q is stale, please manually remove the lock file and retry!",
				installFile, lockPath)
		case <-time.After(RetryLockPeriod):
			continue
		}
	}
}

// Unlock unlocks the file lock for the installPath.
func (l *fileLock) Unlock() error {
	if l == nil {
		return nil
	}
	err := l.flock.Unlock()
	if err != nil {
		err = errors.Wrapf(err, "failed to unlock %q for install file %q: please clean-up the lock manually",
			l.lockPath, l.installFile)
	}
	return err
}
