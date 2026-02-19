package installer

import (
	"os"
	"path/filepath"
	"testing"
	"testing/synctest"

	"github.com/gofrs/flock"
)

func TestCheckInstallOrFileLock(t *testing.T) {
	// Create a temporary directory for testing
	tmpDir, err := os.MkdirTemp("", "flock_test")
	if err != nil {
		t.Fatalf("Failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tmpDir)

	t.Run("AlreadyInstalled", func(t *testing.T) {
		installFile := filepath.Join(tmpDir, "installed_file")
		if err := os.WriteFile(installFile, []byte("data"), 0644); err != nil {
			t.Fatalf("Failed to create dummy install file: %v", err)
		}

		isInstalled, fLock, err := checkInstallOrFileLock(installFile)
		if err != nil {
			t.Errorf("Unexpected error: %v", err)
		}
		if !isInstalled {
			t.Error("Expected isInstalled to be true")
		}
		if fLock != nil {
			t.Error("Expected fLock to be nil")
		}
	})

	t.Run("NotInstalled_AcquireLock", func(t *testing.T) {
		installFile := filepath.Join(tmpDir, "new_file")
		lockFile := installFile + ".lock"

		isInstalled, fLock, err := checkInstallOrFileLock(installFile)
		if err != nil {
			t.Errorf("Unexpected error: %v", err)
		}
		if isInstalled {
			t.Error("Expected isInstalled to be false")
		}
		if fLock == nil {
			t.Fatal("Expected fLock to be non-nil")
		}
		defer fLock.Unlock()

		// Verify lock file exists
		if _, err := os.Stat(lockFile); os.IsNotExist(err) {
			t.Errorf("Lock file %q was not created", lockFile)
		}

		// Verify we actually hold the lock (try to lock functionality)
		testFlock := flock.New(lockFile)
		locked, err := testFlock.TryLock()
		if err != nil {
			t.Fatalf("Failed to check lock status: %v", err)
		}
		if locked {
			t.Error("Expected to fail acquiring lock, but succeeded (meaning strictly exclusive lock wasn't held)")
			testFlock.Unlock()
		}
	})

	t.Run("LockTimeout", func(t *testing.T) {
		synctest.Test(t, func(t *testing.T) {
			installFile := filepath.Join(tmpDir, "time_out_file")
			lockFile := installFile + ".lock"

			// Manually acquire lock first
			l := flock.New(lockFile)
			locked, err := l.TryLock()
			if err != nil {
				t.Fatalf("Failed to acquire initial lock: %v", err)
			}
			if !locked {
				t.Fatal("Could not acquire initial lock")
			}
			defer l.Unlock()

			// Ensure the environment is settled before starting the check.
			synctest.Wait()

			// Try to acquire lock using function - should timeout
			isInstalled, fLock, err := checkInstallOrFileLock(installFile)
			if err == nil {
				t.Error("Expected timeout error, got nil")
				if fLock != nil {
					fLock.Unlock()
				}
			} else {
				// Expected error
				t.Logf("Got expected timeout error: %v", err)
			}

			if isInstalled {
				t.Error("Expected isInstalled to be false")
			}
		})
	})
}
