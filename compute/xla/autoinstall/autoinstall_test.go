// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package autoinstall_test

import (
	"errors"
	"testing"

	"github.com/gomlx/go-xla/compute/xla"
	_ "github.com/gomlx/go-xla/compute/xla/autoinstall"
)

func TestAutoInstallRegistered(t *testing.T) {
	// With the blank import, AutoInstall should be registered,
	// so calling AutoInstall() must not return ErrNoAutoInstaller.
	err := xla.AutoInstall()
	if errors.Is(err, xla.ErrNoAutoInstaller) {
		t.Fatalf("expected auto-installer to be registered, but got ErrNoAutoInstaller: %v", err)
	}
}
