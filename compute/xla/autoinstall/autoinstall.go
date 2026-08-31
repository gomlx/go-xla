// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

// Package autoinstall automatically registers the default PJRT plugin installer into the XLA backend.
//
// To enable automatic downloading and installation of PJRT plugins when using the "xla" backend,
// import this package for its side effects:
//
//	import _ "github.com/gomlx/go-xla/compute/xla/autoinstall"
package autoinstall

import (
	"github.com/gomlx/go-xla/compute/xla"
	"github.com/gomlx/go-xla/installer"
)

type defaultInstaller struct{}

// AutoInstall implements xla.AutoInstaller.
func (defaultInstaller) AutoInstall() error {
	return installer.AutoInstall("", true, installer.Normal)
}

// AutoInstallPlugin implements xla.AutoInstaller.
func (defaultInstaller) AutoInstallPlugin(pluginName string) error {
	return installer.AutoInstallPlugin(pluginName, "", true, installer.Normal)
}

func init() {
	xla.RegisterAutoInstaller(defaultInstaller{})
}
