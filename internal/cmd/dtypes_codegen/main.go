// codegen parses the pjrt_c_api.h and generates boilerplate code for creating the various C structures.
package main

import (
	"bytes"
	"io"
	"os"

	"github.com/gomlx/go-xla/internal/must"
)

const pjrtCAPIHFilePath = "../../../pkg/pjrt/pjrt_c_api.h"

func main() {
	// Read pjrt_c_api.h
	f := must.M1(os.OpenFile(pjrtCAPIHFilePath, os.O_RDONLY, os.ModePerm))
	var b bytes.Buffer
	_ = must.M1(io.Copy(&b, f))
	must.M(f.Close())
	contents := b.String()

	// Create various Go generate files.
	generateEnums(contents)
}
