#!/bin/bash

# Run this from the root of gopjrt repository to generate docs/coverage.out with the coverage data.

PACKAGE_COVERAGE="github.com/gomlx/go-xla/pjrt,github.com/gomlx/go-xla/xlabuilder"
go test -cover -coverprofile docs/coverage.out -coverpkg="${PACKAGE_COVERAGE}" ./... -test.count=1 -test.short
go tool cover -func docs/coverage.out -o docs/coverage.out
