// pjrt_codegen copies prjt_c_api.h from github.com/openxla/xla source (pointed by XLA_SRC env variable),
// parses it and generates boilerplate code for creating the various C structures.
package main

import (
	"log"
	"os"
	"path"

	"github.com/gomlx/go-xla/internal/must"
	"github.com/gomlx/go-xla/pkg/installer"
)

const (
	xlaSrcEnvVar    = "XLA_SRC"
	pjrtAPIFileName = "pjrt_c_api.h"
)

func main() {
	xlaSrc := os.Getenv(xlaSrcEnvVar)
	if xlaSrc == "" {
		log.Fatalf("Please set %s to the directory containing the cloned github.com/openxla/xla repository.\n", xlaSrcEnvVar)
	}
	xlaSrc = must.M1(installer.ReplaceTildeInDir(xlaSrc))

	// Copy pjrt_c_api.h.
	contentsBytes := must.M1(os.ReadFile(path.Join(xlaSrc, "xla", "pjrt", "c", pjrtAPIFileName)))
	must.M(os.WriteFile(pjrtAPIFileName, contentsBytes, 0644))

	// Create various Go generate files.
	contents := string(contentsBytes)
	generateNewStruct(contents)
	generateAPICalls(contents)
	generateEnums(contents)
}
